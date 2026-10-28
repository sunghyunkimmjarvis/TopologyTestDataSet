# ============================================================
# 03_Train_GINE_Edge_Features.py
#
# Excel format produced by 02_toMake_Excel_File_EDGE.py
# ------------------------------------------------------------
# Sheet 1: Node_Matrix
#   Column A              : STEP file name / case name
#   Column B ~ blank col  : node feature matrix + label
#   After blank col       : adjacency matrix
#   Blank row             : separator between graph cases
#
# Sheet 2: Edge_Features
#   filename, source_node, target_node,
#   convexity_code, dihedral_angle_deg, shared_edge_length,
#   edge_type_line, edge_type_arc, edge_type_spline, ...
#
# Model
# ------------------------------------------------------------
# GINEConv uses both node features and edge_attr.
# MLP baseline uses node features only.
# ============================================================

import logging
from datetime import datetime
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv


# ============================================================
# 1. CONFIG — LOFO: train four families, test cantilever
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# These workbooks must contain both Node_Matrix and Edge_Features sheets.
TRAIN_EXCEL_FILES = [
    "Train_Everything_Except_Flange.xlsx",
]
TEST_EXCEL_FILES = [
    "Flange_EDGE_Feat.xlsx",
]

NODE_SHEET = "Node_Matrix"
EDGE_SHEET = "Edge_Features"

NUM_CLASSES = 3
HIDDEN_DIM = 64
DROPOUT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 100
VALID_RATIO = 0.20
RANDOM_SEED = 42

FEATURE_START_COL = 1
FEATURE_END_COL = 13       # exclusive; 11 node features + 1 label
ADJ_START_COL = 14

EDGE_FEATURE_COLUMNS = [
    "convexity_code",
    "dihedral_angle_norm",
    "shared_length_norm",
    "edge_type_line",
    "edge_type_arc",
    "edge_type_spline",
]

MODEL_OUTPUT_FILE = SCRIPT_DIR / "gine_lofo_test_cantilever_edge.pt"

# ============================================================
# 2. LOGGING / SEED
# ============================================================

def setup_experiment_log(experiment_name: str = "GINE_Edge") -> str:
    class TeeStream:
        def __init__(self, original_stream, logger, log_level=logging.INFO):
            self.original_stream = original_stream
            self.logger = logger
            self.log_level = log_level
            self.buffer = ""

        def write(self, message):
            self.original_stream.write(message)
            self.original_stream.flush()
            self.buffer += message
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                if line.strip():
                    self.logger.log(self.log_level, line.rstrip())

        def flush(self):
            self.original_stream.flush()
            if self.buffer.strip():
                self.logger.log(self.log_level, self.buffer.rstrip())
                self.buffer = ""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{timestamp}_{experiment_name}.log")

    logger = logging.getLogger(f"experiment_logger_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    config_lines = [
        "=" * 72,
        "EXPERIMENT CONFIGURATION",
        "=" * 72,
        f"Run time              : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Python file           : {os.path.abspath(__file__)}",
        f"Train Excel files     : {TRAIN_EXCEL_FILES}",
        f"Held-out test files   : {TEST_EXCEL_FILES}",
        f"Model                 : GINEConv",
        f"Edge features         : {EDGE_FEATURE_COLUMNS}",
        f"Hidden dimension      : {HIDDEN_DIM}",
        f"Dropout               : {DROPOUT}",
        f"Batch size            : {BATCH_SIZE}",
        f"Learning rate         : {LEARNING_RATE}",
        f"Weight decay          : {WEIGHT_DECAY}",
        f"Epochs                : {EPOCHS}",
        f"Random seed           : {RANDOM_SEED}",
        f"Internal valid ratio  : {VALID_RATIO}",
        "=" * 72,
    ]
    for line in config_lines:
        logger.info(line)

    sys.stdout = TeeStream(sys.__stdout__, logger, logging.INFO)
    sys.stderr = TeeStream(sys.__stderr__, logger, logging.ERROR)
    print("Log file:", log_path)
    return log_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. EXCEL PARSING
# ============================================================

def normalize_filename(value) -> str:
    """Excel filename matching을 안정화한다."""
    if pd.isna(value):
        return ""
    return os.path.basename(str(value).strip()).lower()


def split_node_blocks(node_df: pd.DataFrame) -> List[pd.DataFrame]:
    blocks: List[pd.DataFrame] = []
    rows = []

    for _, row in node_df.iterrows():
        if row.isna().all():
            if rows:
                blocks.append(pd.DataFrame(rows).reset_index(drop=True))
                rows = []
        else:
            rows.append(row)

    if rows:
        blocks.append(pd.DataFrame(rows).reset_index(drop=True))

    return blocks


def prepare_edge_table(edge_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "filename",
        "source_node",
        "target_node",
        "convexity_code",
        "dihedral_angle_deg",
        "shared_edge_length",
        "edge_type_line",
        "edge_type_arc",
        "edge_type_spline",
    }
    missing = required - set(edge_df.columns)
    if missing:
        raise ValueError(
            "Edge_Features sheet is missing required columns: "
            + ", ".join(sorted(missing))
        )

    result = edge_df.copy()
    result["filename_key"] = result["filename"].map(normalize_filename)

    numeric_columns = [
        "source_node", "target_node", "convexity_code",
        "dihedral_angle_deg", "shared_edge_length",
        "edge_type_line", "edge_type_arc", "edge_type_spline",
    ]
    for col in numeric_columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result["convexity_code"] = result["convexity_code"].fillna(0.0)
    result["dihedral_angle_deg"] = result["dihedral_angle_deg"].fillna(0.0)
    result["shared_edge_length"] = result["shared_edge_length"].fillna(0.0)
    result[["edge_type_line", "edge_type_arc", "edge_type_spline"]] = (
        result[["edge_type_line", "edge_type_arc", "edge_type_spline"]].fillna(0.0)
    )

    return result


def process_one_case(
    block_df: pd.DataFrame,
    case_index: int,
    all_edges_df: pd.DataFrame,
) -> Data:
    block_df = block_df.reset_index(drop=True)
    case_name = normalize_filename(block_df.iloc[0, 0])
    if not case_name:
        case_name = f"case_{case_index}"

    # ---------------- Node features / labels ----------------
    feature_df = block_df.iloc[:, FEATURE_START_COL:FEATURE_END_COL].copy()
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

    if feature_df.isna().any().any():
        bad = np.argwhere(feature_df.isna().to_numpy())
        raise ValueError(
            f"NaN/non-numeric node feature in {case_name}; first bad position={bad[0].tolist()}"
        )

    x_df = feature_df.iloc[:, :-1]
    y_df = feature_df.iloc[:, -1]

    x = torch.tensor(x_df.to_numpy(dtype=np.float32), dtype=torch.float)
    y = torch.tensor(y_df.to_numpy(dtype=np.int64), dtype=torch.long)
    num_nodes = x.size(0)

    # ---------------- Adjacency ----------------
    available_adj_cols = max(0, block_df.shape[1] - ADJ_START_COL)
    read_adj_cols = min(num_nodes, available_adj_cols)
    raw_adj = (
        block_df.iloc[:num_nodes, ADJ_START_COL:ADJ_START_COL + read_adj_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
    )
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    adj[:, :read_adj_cols] = raw_adj
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)

    # ---------------- Edge attributes ----------------
    case_edges = all_edges_df[all_edges_df["filename_key"] == case_name].copy()
    if case_edges.empty:
        raise ValueError(
            f"No edge rows found for '{case_name}' in sheet '{EDGE_SHEET}'."
        )

    max_length = float(case_edges["shared_edge_length"].max())
    if not np.isfinite(max_length) or max_length <= 0.0:
        max_length = 1.0

    directed_edges: List[List[int]] = []
    directed_attrs: List[List[float]] = []
    seen_pairs = set()

    for _, edge in case_edges.iterrows():
        if pd.isna(edge["source_node"]) or pd.isna(edge["target_node"]):
            continue

        source = int(edge["source_node"])
        target = int(edge["target_node"])

        if not (0 <= source < num_nodes and 0 <= target < num_nodes):
            raise ValueError(
                f"Invalid edge node index in {case_name}: {source}->{target}, "
                f"num_nodes={num_nodes}"
            )
        if source == target:
            continue

        attr = [
            float(edge["convexity_code"]),
            float(edge["dihedral_angle_deg"]) / 180.0,
            float(edge["shared_edge_length"]) / max_length,
            float(edge["edge_type_line"]),
            float(edge["edge_type_arc"]),
            float(edge["edge_type_spline"]),
        ]

        # Excel에는 undirected shared edge가 1행이므로 PyG용 양방향 생성
        directed_edges.extend([[source, target], [target, source]])
        directed_attrs.extend([attr, attr])
        seen_pairs.add(tuple(sorted((source, target))))

    if not directed_edges:
        raise ValueError(f"No valid directed edges were constructed for {case_name}.")

    # Node_Matrix adjacency와 Edge_Features의 연결관계 일치 여부 검사
    adjacency_pairs = {
        (i, j)
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
        if adj[i, j] == 1
    }
    missing_feature_pairs = adjacency_pairs - seen_pairs
    extra_feature_pairs = seen_pairs - adjacency_pairs

    if missing_feature_pairs:
        raise ValueError(
            f"{case_name}: adjacency edges without edge features: "
            f"{sorted(missing_feature_pairs)[:10]}"
        )
    if extra_feature_pairs:
        print(
            f"WARNING {case_name}: Edge_Features contains connections absent from "
            f"Node_Matrix adjacency: {sorted(extra_feature_pairs)[:10]}"
        )

    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(directed_attrs, dtype=torch.float)

    if edge_index.size(1) != edge_attr.size(0):
        raise RuntimeError(
            f"edge_index/edge_attr mismatch in {case_name}: "
            f"{edge_index.size(1)} != {edge_attr.size(0)}"
        )
    if not torch.isfinite(edge_attr).all():
        raise ValueError(f"Non-finite edge attributes found in {case_name}.")

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.case_name = case_name
    data.case_index = case_index
    return data


def load_dataset_from_excel(excel_file: str) -> List[Data]:
    print("\nLoading:", excel_file)

    node_df = pd.read_excel(excel_file, sheet_name=NODE_SHEET, header=None)
    raw_edge_df = pd.read_excel(excel_file, sheet_name=EDGE_SHEET, header=0)


    print("Reading Excel:", excel_file)
    print("Edge columns:", raw_edge_df.columns.tolist())
    
    edge_df = prepare_edge_table(raw_edge_df)

    print(raw_edge_df.head())
    print(raw_edge_df.columns.tolist())

    print(raw_edge_df.columns.tolist())   # 추가

    print("Node_Matrix shape:", node_df.shape)
    print("Edge_Features shape:", edge_df.shape)

    blocks = split_node_blocks(node_df)
    dataset = [
        process_one_case(block, i, edge_df)
        for i, block in enumerate(blocks)
    ]

    if not dataset:
        raise ValueError("No graph case was loaded from Excel.")

    print("Dataset size:", len(dataset))
    print("Node feature dimension:", dataset[0].num_node_features)
    print("Edge feature dimension:", dataset[0].edge_attr.size(1))
    print("First graph:", dataset[0])
    return dataset


def resolve_excel_path(file_name: str) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    search_paths = [
        SCRIPT_DIR / file_name,
        SCRIPT_DIR / "TestDataResult_Excel" / file_name,
        Path.cwd() / file_name,
        Path.cwd() / "TestDataResult_Excel" / file_name,
    ]
    for path in search_paths:
        if path.exists():
            return path.resolve()
    checked = "\n".join(f"  - {p}" for p in search_paths)
    raise FileNotFoundError(f"Excel file not found: {file_name}\nChecked:\n{checked}")


def load_multiple_excels(file_names: Sequence[str]) -> List[Data]:
    combined: List[Data] = []
    for file_name in file_names:
        path = resolve_excel_path(file_name)
        part = load_dataset_from_excel(str(path))
        for data in part:
            data.source_file = path.name
        combined.extend(part)
    return combined


# ============================================================
# 4. DATASET SPLIT
# ============================================================

def split_train_validation(
    dataset: List[Data],
    valid_ratio: float,
    seed: int,
) -> Tuple[List[Data], List[Data]]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("VALID_RATIO must be between 0 and 1.")
    if len(dataset) < 2:
        raise ValueError("At least two seen-family graphs are required.")
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    valid_size = max(1, int(round(len(indices) * valid_ratio)))
    valid_size = min(valid_size, len(indices) - 1)
    valid_idx = indices[:valid_size]
    train_idx = indices[valid_size:]
    return [dataset[i] for i in train_idx], [dataset[i] for i in valid_idx]


# ============================================================
# 5. MODELS
# ============================================================

class GINENodeClassifier(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)

        mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv1 = GINEConv(mlp1, edge_dim=edge_feature_dim, train_eps=True)
        self.conv2 = GINEConv(mlp2, edge_dim=edge_feature_dim, train_eps=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)

        residual = x
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x + residual)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None, edge_attr=None):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.classifier(x)


# ============================================================
# 6. TRAIN / EVALUATION
# ============================================================

def forward_model(model, batch):
    if isinstance(model, GINENodeClassifier):
        return model(batch.x, batch.edge_index, batch.edge_attr)
    return model(batch.x, batch.edge_index, batch.edge_attr)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = forward_model(model, batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def test_node_level(model, loader, device):
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        batch = batch.to(device)
        out = forward_model(model, batch)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())

    if not all_labels:
        raise ValueError("Evaluation loader contains no nodes.")

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    return acc, cm


@torch.no_grad()
def test_case_level(model, loader, device):
    model.eval()
    total_cases = 0
    correct_cases = 0

    for batch in loader:
        batch = batch.to(device)
        out = forward_model(model, batch)
        pred = out.argmax(dim=1)

        for i in range(len(batch.ptr) - 1):
            start = int(batch.ptr[i])
            end = int(batch.ptr[i + 1])
            total_cases += 1
            if torch.equal(pred[start:end], batch.y[start:end]):
                correct_cases += 1

    if total_cases == 0:
        raise ValueError("Evaluation loader contains no graph cases.")

    return correct_cases / total_cases, correct_cases, total_cases


def clone_state_dict(model) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


# ============================================================
# 7. MAIN
# ============================================================

def main():
    setup_experiment_log("GINE_LOFO_Edge_Features")
    set_seed(RANDOM_SEED)

    # Held-out cantilever is never used for model selection.
    seen_family_dataset = load_multiple_excels(TRAIN_EXCEL_FILES)
    held_out_test_dataset = load_multiple_excels(TEST_EXCEL_FILES)
    train_dataset, valid_dataset = split_train_validation(
        seen_family_dataset, VALID_RATIO, RANDOM_SEED
    )

    node_feature_dim = train_dataset[0].num_node_features
    edge_feature_dim = train_dataset[0].edge_attr.size(1)

    for name, dataset in (
        ("training", train_dataset),
        ("validation", valid_dataset),
        ("held-out test", held_out_test_dataset),
    ):
        node_dims = {d.num_node_features for d in dataset}
        edge_dims = {d.edge_attr.size(1) for d in dataset}
        if node_dims != {node_feature_dim}:
            raise ValueError(f"Inconsistent node dimensions in {name}: {node_dims}")
        if edge_dims != {edge_feature_dim}:
            raise ValueError(f"Inconsistent edge dimensions in {name}: {edge_dims}")

    print("\n================ DATASET SUMMARY ================")
    print("Seen-family total cases       :", len(seen_family_dataset))
    print("Training cases                :", len(train_dataset))
    print("Internal validation cases     :", len(valid_dataset))
    print("Held-out Target test cases:", len(held_out_test_dataset))
    print("Number of node features       :", node_feature_dim)
    print("Number of edge features       :", edge_feature_dim)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(held_out_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = GINENodeClassifier(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    best_epoch = 0
    best_node_acc = -1.0
    best_case_acc = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_node_acc, valid_cm = test_node_level(model, valid_loader, device)
        valid_case_acc, _, _ = test_case_level(model, valid_loader, device)

        is_better = (
            valid_node_acc > best_node_acc
            or (valid_node_acc == best_node_acc and valid_case_acc > best_case_acc)
        )
        if is_better:
            best_epoch = epoch
            best_node_acc = valid_node_acc
            best_case_acc = valid_case_acc
            best_state = clone_state_dict(model)

        print(
            f"Epoch {epoch:04d} | Train Loss {train_loss:.6f} | "
            f"Valid Node Acc {valid_node_acc:.4f} | "
            f"Valid Geometry Acc {valid_case_acc:.4f}"
        )

    if best_state is None:
        raise RuntimeError("No best model checkpoint was created.")
    model.load_state_dict(best_state)

    # Final held-out evaluation occurs once, after model selection.
    test_node_acc, test_cm = test_node_level(model, test_loader, device)
    test_case_acc, correct_cases, total_cases = test_case_level(
        model, test_loader, device
    )

    print("\n================ BEST VALIDATION ================")
    print("Best epoch                  :", best_epoch)
    print(f"Best validation node acc    : {best_node_acc:.4f}")
    print(f"Best validation geometry acc: {best_case_acc:.4f}")

    print("\n========== HELD-OUT CANTILEVER TEST =============")
    print(f"Node-level accuracy         : {test_node_acc:.4f}")
    print(
        f"Case-level strict accuracy  : {correct_cases} / {total_cases} "
        f"= {test_case_acc:.4f}"
    )
    print("\nNode-level Confusion Matrix")
    print(test_cm)

    torch.save(
        {
            "model_state_dict": best_state,
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "edge_feature_columns": EDGE_FEATURE_COLUMNS,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": NUM_CLASSES,
            "best_epoch": best_epoch,
            "best_validation_node_accuracy": best_node_acc,
            "best_validation_case_accuracy": best_case_acc,
            "train_files": TRAIN_EXCEL_FILES,
            "test_files": TEST_EXCEL_FILES,
            "random_seed": RANDOM_SEED,
        },
        MODEL_OUTPUT_FILE,
    )
    print("Saved best model:", MODEL_OUTPUT_FILE)


if __name__ == "__main__":
    main()
