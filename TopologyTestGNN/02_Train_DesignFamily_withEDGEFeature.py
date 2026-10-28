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
from typing import Dict, List, Optional, Tuple
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
# 1. CONFIG
# ============================================================
EXCEL_FILE = r"TestDataResult_Excel\BookShelves_EDGE_Feat.xlsx"  # Change the target to cantilever, Bolt, BookShelves, Flange, or LshapedBracket.
VALIDATION_EXCEL_FILE = None
NODE_SHEET = "Node_Matrix"
EDGE_SHEET = "Edge_Features"
NUM_CLASSES = 3
HIDDEN_DIM = 64
DROPOUT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 100
RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2
# Existing Excel Node_Matrix structure
FEATURE_START_COL = 1
FEATURE_END_COL = 13       # exclusive; 12 columns = 11 node features + label
ADJ_START_COL = 14
# Order of edge attributes supplied to GINEConv
EDGE_FEATURE_COLUMNS = [
    "convexity_code",       # CONCAVE=-1, UNKNOWN/SMOOTH=0, CONVEX=1
    "dihedral_angle_norm",  # dihedral_angle_deg / 180
    "shared_length_norm",   # Normalize by the maximum shared-edge length in each graph
    "edge_type_line",
    "edge_type_arc",
    "edge_type_spline",
]
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
        f"Excel file            : {EXCEL_FILE}",
        f"Validation Excel file : {VALIDATION_EXCEL_FILE}",
        f"Model                 : GINEConv",
        f"Edge features         : {EDGE_FEATURE_COLUMNS}",
        f"Hidden dimension      : {HIDDEN_DIM}",
        f"Dropout               : {DROPOUT}",
        f"Batch size            : {BATCH_SIZE}",
        f"Learning rate         : {LEARNING_RATE}",
        f"Weight decay          : {WEIGHT_DECAY}",
        f"Epochs                : {EPOCHS}",
        f"Random seed           : {RANDOM_SEED}",
        f"Split                 : {TRAIN_RATIO}/{VALID_RATIO}/{TEST_RATIO}",
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
    """Normalize Excel filenames for reliable matching."""
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
        # Each undirected shared edge occupies one Excel row; create both directions for PyG.
        directed_edges.extend([[source, target], [target, source]])
        directed_attrs.extend([attr, attr])
        seen_pairs.add(tuple(sorted((source, target))))
    if not directed_edges:
        raise ValueError(f"No valid directed edges were constructed for {case_name}.")
    # Verify connectivity consistency between Node_Matrix and Edge_Features.
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
    edge_df = prepare_edge_table(raw_edge_df)
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
# ============================================================
# 4. DATASET SPLIT
# ============================================================
def split_dataset(
    dataset: List[Data],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Tuple[List[Data], List[Data], List[Data]]:
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Train/validation/test ratios must sum to 1.0")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    total = len(indices)
    train_size = int(total * train_ratio)
    valid_size = int(total * valid_ratio)
    test_size = total - train_size - valid_size
    invalid = (
        (train_ratio > 0 and train_size == 0)
        or (valid_ratio > 0 and valid_size == 0)
        or (test_ratio > 0 and test_size == 0)
    )
    if invalid:
        raise ValueError(
            f"Dataset too small for split: total={total}, "
            f"train={train_size}, valid={valid_size}, test={test_size}"
        )
    train_idx = indices[:train_size]
    valid_idx = indices[train_size:train_size + valid_size]
    test_idx = indices[train_size + valid_size:]
    return (
        [dataset[i] for i in train_idx],
        [dataset[i] for i in valid_idx],
        [dataset[i] for i in test_idx],
    )
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
    setup_experiment_log("GINE_Edge_Features")
    set_seed(RANDOM_SEED)
    if VALIDATION_EXCEL_FILE is None:
        full_dataset = load_dataset_from_excel(EXCEL_FILE)
        train_dataset, valid_dataset, test_dataset = split_dataset(
            full_dataset, TRAIN_RATIO, VALID_RATIO, TEST_RATIO
        )
    else:
        full_train_dataset = load_dataset_from_excel(EXCEL_FILE)
        train_dataset, valid_dataset, _ = split_dataset(
            full_train_dataset,
            TRAIN_RATIO / (TRAIN_RATIO + VALID_RATIO),
            VALID_RATIO / (TRAIN_RATIO + VALID_RATIO),
            0.0,
        )
        test_dataset = load_dataset_from_excel(VALIDATION_EXCEL_FILE)
    print("\nTrain cases:", len(train_dataset))
    print("Validation cases:", len(valid_dataset))
    print("Test cases:", len(test_dataset))
    node_feature_dim = train_dataset[0].num_node_features
    edge_feature_dim = train_dataset[0].edge_attr.size(1)
    for data in train_dataset + valid_dataset + test_dataset:
        if data.num_node_features != node_feature_dim:
            raise ValueError("Inconsistent node feature dimensions across graphs.")
        if data.edge_attr.size(1) != edge_feature_dim:
            raise ValueError("Inconsistent edge feature dimensions across graphs.")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    gine_model = GINENodeClassifier(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)
    mlp_model = MLPBaseline(
        input_dim=node_feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=0.0,
    ).to(device)
    gine_optimizer = torch.optim.Adam(
        gine_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    mlp_optimizer = torch.optim.Adam(
        mlp_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    best_gine_acc = -1.0
    best_mlp_acc = -1.0
    best_gine_state: Optional[Dict[str, torch.Tensor]] = None
    best_mlp_state: Optional[Dict[str, torch.Tensor]] = None
    best_gine_epoch = 0
    best_mlp_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        gine_loss = train_one_epoch(
            gine_model, train_loader, gine_optimizer, criterion, device
        )
        mlp_loss = train_one_epoch(
            mlp_model, train_loader, mlp_optimizer, criterion, device
        )
        gine_val_acc, _ = test_node_level(gine_model, valid_loader, device)
        mlp_val_acc, _ = test_node_level(mlp_model, valid_loader, device)
        gine_case_acc, _, _ = test_case_level(gine_model, valid_loader, device)
        mlp_case_acc, _, _ = test_case_level(mlp_model, valid_loader, device)
        if gine_val_acc > best_gine_acc:
            best_gine_acc = gine_val_acc
            best_gine_epoch = epoch
            best_gine_state = clone_state_dict(gine_model)
        if mlp_val_acc > best_mlp_acc:
            best_mlp_acc = mlp_val_acc
            best_mlp_epoch = epoch
            best_mlp_state = clone_state_dict(mlp_model)
        print(
            f"Epoch {epoch:04d} | "
            f"GINE Loss {gine_loss:.6f} | GINE Node Acc {gine_val_acc:.4f} | "
            f"GINE Geometry Acc {gine_case_acc:.4f} | "
            f"MLP Loss {mlp_loss:.6f} | MLP Node Acc {mlp_val_acc:.4f} | "
            f"MLP Geometry Acc {mlp_case_acc:.4f}"
        )
    if best_gine_state is None or best_mlp_state is None:
        raise RuntimeError("Best model state was not created.")
    gine_model.load_state_dict(best_gine_state)
    mlp_model.load_state_dict(best_mlp_state)
    print("\n" + "=" * 72)
    print("BEST VALIDATION CHECKPOINTS")
    print("=" * 72)
    print(f"GINE best epoch: {best_gine_epoch}, node accuracy: {best_gine_acc:.4f}")
    print(f"MLP  best epoch: {best_mlp_epoch}, node accuracy: {best_mlp_acc:.4f}")
    gine_val_acc, gine_val_cm = test_node_level(gine_model, valid_loader, device)
    mlp_val_acc, mlp_val_cm = test_node_level(mlp_model, valid_loader, device)
    print("\nGINE Validation Node Accuracy:", gine_val_acc)
    print("GINE Validation Confusion Matrix")
    print(gine_val_cm)
    print("\nMLP Validation Node Accuracy:", mlp_val_acc)
    print("MLP Validation Confusion Matrix")
    print(mlp_val_cm)
    gine_test_acc, gine_test_cm = test_node_level(gine_model, test_loader, device)
    mlp_test_acc, mlp_test_cm = test_node_level(mlp_model, test_loader, device)
    gine_case_acc, gine_correct, gine_total = test_case_level(
        gine_model, test_loader, device
    )
    mlp_case_acc, mlp_correct, mlp_total = test_case_level(
        mlp_model, test_loader, device
    )
    print("\n" + "=" * 72)
    print("FINAL TEST RESULTS")
    print("=" * 72)
    print(f"GINE Test Node Accuracy: {gine_test_acc:.4f}")
    print("GINE Test Confusion Matrix")
    print(gine_test_cm)
    print(
        f"GINE Case-level Strict Accuracy: "
        f"{gine_correct} / {gine_total} = {gine_case_acc:.4f}"
    )
    print(f"\nMLP Test Node Accuracy: {mlp_test_acc:.4f}")
    print("MLP Test Confusion Matrix")
    print(mlp_test_cm)
    print(
        f"MLP Case-level Strict Accuracy: "
        f"{mlp_correct} / {mlp_total} = {mlp_case_acc:.4f}"
    )
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "best_gine_edge_model.pt",
    )
    torch.save(
        {
            "model_state_dict": gine_model.state_dict(),
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "edge_feature_columns": EDGE_FEATURE_COLUMNS,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": NUM_CLASSES,
            "best_epoch": best_gine_epoch,
            "best_validation_node_accuracy": best_gine_acc,
        },
        checkpoint_path,
    )
    print("\nSaved GINE checkpoint:", checkpoint_path)
if __name__ == "__main__":
    main()
