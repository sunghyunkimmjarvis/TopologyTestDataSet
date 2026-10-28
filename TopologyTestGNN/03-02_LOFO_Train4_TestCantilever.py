# ============================================================
# 03_LOFO_Train4_TestCantilever.py
#
# Leave-One-Family-Out (LOFO) experiment
# ------------------------------------------------------------
# Train families : Bolt + Shelves + Flange + L-Shaped Bracket
# Held-out family: Cantilever Beam
#
# Excel format (0-based pandas iloc indices)
# ------------------------------------------------------------
# A column         : case/file name
# B ~ L columns    : 11 node features
# M column         : node label (0, 1, 2)
# N column         : blank separator column
# O onward         : adjacency matrix
# Blank row        : separator between graph cases
# ============================================================
import logging
from datetime import datetime
import os
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import numpy as np

import subprocess
import sys


def setup_experiment_log(experiment_name="experiment"):
    """
    실행할 때마다 로그 파일을 생성하고 다음 내용을 저장한다.

    1. 실행 시각
    2. Python 파일 경로
    3. 실험 CONFIG
    4. 이후 발생하는 모든 print() 출력
    5. 오류 메시지(stderr)

    콘솔 출력과 .log 파일 기록이 동시에 이루어진다.
    """

    # ---------------------------------------------------------
    # print 출력을 콘솔과 로그 파일에 동시에 보내는 내부 클래스
    # ---------------------------------------------------------
    class TeeStream:

        def __init__(
            self,
            original_stream,
            logger,
            log_level=logging.INFO
        ):
            self.original_stream = original_stream
            self.logger = logger
            self.log_level = log_level
            self.buffer = ""

        def write(self, message):

            # 기존 콘솔 출력 유지
            self.original_stream.write(message)
            self.original_stream.flush()

            # print()가 여러 조각으로 전달되는 경우를 고려
            self.buffer += message

            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)

                if line.strip():
                    self.logger.log(
                        self.log_level,
                        line.rstrip()
                    )

        def flush(self):

            self.original_stream.flush()

            # 버퍼에 남은 내용 기록
            if self.buffer.strip():
                self.logger.log(
                    self.log_level,
                    self.buffer.rstrip()
                )
                self.buffer = ""

    # ---------------------------------------------------------
    # logs 폴더와 로그 파일 이름 생성
    # ---------------------------------------------------------
    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    log_dir = os.path.join(
        script_dir,
        "logs"
    )

    os.makedirs(
        log_dir,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_name = (
        f"{timestamp}_{experiment_name}.log"
    )

    log_path = os.path.join(
        log_dir,
        log_name
    )

    # ---------------------------------------------------------
    # Logger 생성
    # ---------------------------------------------------------
    logger = logging.getLogger(
        f"experiment_logger_{timestamp}"
    )

    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 중복 handler 방지
    logger.handlers.clear()

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8"
    )

    file_handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    logger.addHandler(file_handler)

    # ---------------------------------------------------------
    # CONFIG 기록
    # ---------------------------------------------------------
    logger.info("=" * 70)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 70)

    logger.info(
        f"Run time              : "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    logger.info(
        f"Experiment name       : {experiment_name}"
    )

    logger.info(
        f"Python file           : "
        f"{os.path.abspath(__file__)}"
    )

    logger.info(
        f"Excel file            : {EXCEL_FILE}"
    )

    logger.info(
        f"Validation Excel file : "
        f"{VALIDATION_EXCEL_FILE}"
    )

    logger.info(
        f"Number of classes     : {NUM_CLASSES}"
    )

    logger.info(
        f"Batch size            : {BATCH_SIZE}"
    )

    logger.info(
        f"Learning rate         : {LEARNING_RATE}"
    )

    logger.info(
        f"Epochs                : {EPOCHS}"
    )

    logger.info(
        f"Random seed           : {RANDOM_SEED}"
    )

    logger.info(
        f"Train ratio           : {TRAIN_RATIO}"
    )

    logger.info(
        f"Validation ratio      : {VALID_RATIO}"
    )

    logger.info(
        f"Test ratio            : {TEST_RATIO}"
    )

    logger.info("=" * 70)
    logger.info("")

    # ---------------------------------------------------------
    # 이후의 모든 print 및 오류 출력을 로그에 연결
    # ---------------------------------------------------------
    sys.stdout = TeeStream(
        original_stream=sys.__stdout__,
        logger=logger,
        log_level=logging.INFO
    )

    sys.stderr = TeeStream(
        original_stream=sys.__stderr__,
        logger=logger,
        log_level=logging.ERROR
    )

    print("Log file:", log_path)
    print()

    return log_path

# ============================================================
# 1. CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Four seen families used for training/model selection.
TRAIN_EXCEL_FILES = [
    "Train_BoltShelvesFlangeLBrack.xlsx",
]

# The entire cantilever family is held out.
# Existing input/validation files are combined only for the FINAL test.
TEST_EXCEL_FILES = [
    "cantilever.xlsx",
]

NUM_CLASSES = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
EPOCHS = 100
VALID_RATIO = 0.20
RANDOM_SEED = 42

# Excel column positions for the uploaded format.
# iloc slicing excludes the end index.
FEATURE_START_COL = 1   # B
FEATURE_END_COL = 13    # B:M -> 12 columns = 11 features + 1 label
ADJ_START_COL = 14      # O

MODEL_OUTPUT_FILE = SCRIPT_DIR / "gcn_lofo_test_cantilever.pt"


# ============================================================
# 2. REPRODUCIBILITY / PATH
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # More reproducible CUDA behavior where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_excel_path(file_name: str) -> Path:
    """Find an Excel file next to the script or in TestDataResult_Excel."""
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

    checked = "\n".join(f"  - {path}" for path in search_paths)
    raise FileNotFoundError(
        f"Excel file was not found: {file_name}\nChecked:\n{checked}"
    )


# ============================================================
# 3. ADJACENCY MATRIX -> EDGE_INDEX
# ============================================================

def adjacency_to_edge_index(adj) -> torch.Tensor:
    """Convert a dense 0/1 adjacency matrix to PyG edge_index."""
    rows, cols = (adj == 1).nonzero()

    if len(rows) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor([rows, cols], dtype=torch.long)


# ============================================================
# 4. ONE EXCEL BLOCK -> PYG DATA
# ============================================================

def process_one_case(block_df: pd.DataFrame, case_index: int) -> Data:
    block_df = block_df.reset_index(drop=True)
    num_nodes = len(block_df)

    if num_nodes == 0:
        raise ValueError(f"Case {case_index}: empty graph block.")

    # B:M = 11 input features + label.
    feature_label_df = block_df.iloc[:, FEATURE_START_COL:FEATURE_END_COL]

    expected_columns = FEATURE_END_COL - FEATURE_START_COL
    if feature_label_df.shape[1] != expected_columns:
        raise ValueError(
            f"Case {case_index}: expected {expected_columns} feature/label "
            f"columns, but found {feature_label_df.shape[1]}."
        )

    # First 11 columns are x; final M column is y.
    x_df = feature_label_df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    y_df = pd.to_numeric(feature_label_df.iloc[:, -1], errors="coerce")

    if x_df.isna().any().any():
        bad = x_df.isna().stack()
        bad_positions = list(bad[bad].index[:5])
        raise ValueError(
            f"Case {case_index}: non-numeric or missing feature values at "
            f"{bad_positions}."
        )

    if y_df.isna().any():
        raise ValueError(f"Case {case_index}: label contains NaN/non-numeric value.")

    labels = y_df.astype(int)
    invalid_labels = sorted(set(labels.tolist()) - set(range(NUM_CLASSES)))
    if invalid_labels:
        raise ValueError(
            f"Case {case_index}: invalid labels {invalid_labels}; "
            f"expected 0..{NUM_CLASSES - 1}."
        )

        # O onward: adjacency matrix.
    # Some Excel files may omit trailing all-empty adjacency columns.
    available_adj_cols = max(
        0,
        block_df.shape[1] - ADJ_START_COL
    )

    read_adj_cols = min(
        num_nodes,
        available_adj_cols
    )

    if read_adj_cols == 0:
        raise ValueError(
            f"Case {case_index}: no adjacency columns were found. "
            f"block columns={block_df.shape[1]}, "
            f"ADJ_START_COL={ADJ_START_COL}."
        )

    adj_df = block_df.iloc[
        :num_nodes,
        ADJ_START_COL:ADJ_START_COL + read_adj_cols,
    ]

    raw_adj = (
        adj_df.apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
    )

    # Always create a square num_nodes x num_nodes matrix.
    adj = np.zeros(
        (num_nodes, num_nodes),
        dtype=int,
    )

    # Copy the adjacency columns that exist in Excel.
    adj[:, :read_adj_cols] = raw_adj

    if read_adj_cols < num_nodes:
        print(
            f"[Warning] Case {case_index}: adjacency shape read as "
            f"{raw_adj.shape}; expected ({num_nodes}, {num_nodes}). "
            f"Reconstructing {num_nodes - read_adj_cols} missing columns "
            f"using matrix symmetry."
        )

    # Restore missing columns using undirected-graph symmetry.
    adj = np.maximum(adj, adj.T)

    # Remove self-loops.
    np.fill_diagonal(adj, 0)

    invalid_adj_values = sorted(set(adj.ravel().tolist()) - {0, 1})
    if invalid_adj_values:
        raise ValueError(
            f"Case {case_index}: adjacency contains values other than 0/1: "
            f"{invalid_adj_values[:10]}"
        )

    case_name = str(block_df.iloc[0, 0])

    data = Data(
        x=torch.tensor(x_df.to_numpy(), dtype=torch.float),
        edge_index=adjacency_to_edge_index(adj),
        y=torch.tensor(labels.to_numpy(), dtype=torch.long),
    )
    data.case_name = case_name
    data.case_index = case_index

    return data


# ============================================================
# 5. EXCEL -> DATASET
# ============================================================

def load_dataset_from_excel(excel_file: Path) -> List[Data]:
    print(f"\nLoading: {excel_file}")

    df = pd.read_excel(excel_file, header=None)
    print("Excel shape:", df.shape)

    dataset: List[Data] = []
    current_rows = []

    for _, row in df.iterrows():
        # Completely blank row separates graph cases.
        if row.isna().all():
            if current_rows:
                block_df = pd.DataFrame(current_rows)
                dataset.append(process_one_case(block_df, len(dataset)))
                current_rows = []
        else:
            current_rows.append(row)

    if current_rows:
        block_df = pd.DataFrame(current_rows)
        dataset.append(process_one_case(block_df, len(dataset)))

    if not dataset:
        raise ValueError(f"No graph case was loaded from {excel_file}.")

    node_counts = [data.num_nodes for data in dataset]
    print("Graph cases:", len(dataset))
    print("Node count range:", min(node_counts), "~", max(node_counts))
    print("Node features:", dataset[0].num_node_features)

    return dataset


def load_multiple_excels(file_names: Sequence[str]) -> List[Data]:
    combined: List[Data] = []

    for file_name in file_names:
        path = resolve_excel_path(file_name)
        file_dataset = load_dataset_from_excel(path)

        # Preserve source workbook for result tracing.
        for data in file_dataset:
            data.source_file = path.name

        combined.extend(file_dataset)

    return combined


# ============================================================
# 6. INTERNAL TRAIN / VALIDATION SPLIT
# ============================================================

def split_train_validation(
    dataset: Sequence[Data],
    valid_ratio: float,
    seed: int,
) -> Tuple[List[Data], List[Data]]:
    """Split only the seen-family data; held-out cantilever stays untouched."""
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("VALID_RATIO must be between 0 and 1.")

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    valid_size = max(1, int(round(len(indices) * valid_ratio)))
    valid_size = min(valid_size, len(indices) - 1)

    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]

    train_dataset = [dataset[i] for i in train_indices]
    valid_dataset = [dataset[i] for i in valid_indices]

    return train_dataset, valid_dataset


# ============================================================
# 7. MODEL
# ============================================================

class GCN(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.linear = torch.nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.linear(x)


# ============================================================
# 8. TRAIN / EVALUATION
# ============================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()

        nodes = int(batch.y.numel())
        total_loss += float(loss.item()) * nodes
        total_nodes += nodes

    return total_loss / max(total_nodes, 1)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    total_cases = 0
    correct_cases = 0
    incorrect_case_names: List[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index).argmax(dim=1)

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

            ptr = batch.ptr.cpu().tolist()
            case_names = getattr(batch, "case_name", None)

            for graph_idx in range(len(ptr) - 1):
                start, end = ptr[graph_idx], ptr[graph_idx + 1]
                is_correct = torch.equal(
                    pred[start:end].cpu(),
                    batch.y[start:end].cpu(),
                )

                total_cases += 1
                if is_correct:
                    correct_cases += 1
                elif case_names is not None:
                    incorrect_case_names.append(str(case_names[graph_idx]))

    if not all_labels:
        raise ValueError("Evaluation loader contained no nodes.")

    node_accuracy = sum(
        int(p == y) for p, y in zip(all_preds, all_labels)
    ) / len(all_labels)

    case_accuracy = correct_cases / max(total_cases, 1)

    return {
        "node_accuracy": node_accuracy,
        "case_accuracy": case_accuracy,
        "correct_cases": correct_cases,
        "total_cases": total_cases,
        "confusion_matrix": confusion_matrix(
            all_labels,
            all_preds,
            labels=list(range(NUM_CLASSES)),
        ),
        "classification_report": classification_report(
            all_labels,
            all_preds,
            labels=list(range(NUM_CLASSES)),
            digits=4,
            zero_division=0,
        ),
        "incorrect_case_names": incorrect_case_names,
    }


# ============================================================
# 9. MAIN: TRAIN ON 4 FAMILIES, FINAL TEST ON CANTILEVER
# ============================================================

def main() -> None:
    set_seed(RANDOM_SEED)

    seen_family_dataset = load_multiple_excels(TRAIN_EXCEL_FILES)
    held_out_test_dataset = load_multiple_excels(TEST_EXCEL_FILES)

    train_dataset, valid_dataset = split_train_validation(
        seen_family_dataset,
        VALID_RATIO,
        RANDOM_SEED,
    )

    num_features = train_dataset[0].num_node_features

    for name, dataset in (
        ("training", train_dataset),
        ("validation", valid_dataset),
        ("held-out test", held_out_test_dataset),
    ):
        feature_counts = {data.num_node_features for data in dataset}
        if feature_counts != {num_features}:
            raise ValueError(
                f"Inconsistent feature count in {name} dataset: {feature_counts}; "
                f"training expects {num_features}."
            )

    print("\n================ DATASET SUMMARY ================")
    print("Seen-family total cases       :", len(seen_family_dataset))
    print("Training cases                :", len(train_dataset))
    print("Internal validation cases     :", len(valid_dataset))
    print("Held-out cantilever test cases:", len(held_out_test_dataset))
    print("Number of node features       :", num_features)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        held_out_test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = GCN(num_features, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Model selection uses internal validation only, never cantilever test data.
    best_epoch = 0
    best_valid_node_acc = -1.0
    best_valid_case_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        valid_result = evaluate(model, valid_loader, device)

        valid_node_acc = valid_result["node_accuracy"]
        valid_case_acc = valid_result["case_accuracy"]

        # Primary selection metric: node accuracy.
        # Case accuracy breaks ties.
        is_better = (
            valid_node_acc > best_valid_node_acc
            or (
                valid_node_acc == best_valid_node_acc
                and valid_case_acc > best_valid_case_acc
            )
        )

        if is_better:
            best_epoch = epoch
            best_valid_node_acc = valid_node_acc
            best_valid_case_acc = valid_case_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:04d} | "
            f"Train Loss {train_loss:.6f} | "
            f"Valid Node Acc {valid_node_acc:.4f} | "
            f"Valid Geometry Acc {valid_case_acc:.4f}"
        )

    if best_state is None:
        raise RuntimeError("No best model checkpoint was created.")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "num_features": num_features,
            "num_classes": NUM_CLASSES,
            "best_epoch": best_epoch,
            "best_valid_node_accuracy": best_valid_node_acc,
            "best_valid_case_accuracy": best_valid_case_acc,
            "train_files": TRAIN_EXCEL_FILES,
            "test_files": TEST_EXCEL_FILES,
            "random_seed": RANDOM_SEED,
        },
        MODEL_OUTPUT_FILE,
    )

    # Cantilever is evaluated exactly once after model selection.
    test_result = evaluate(model, test_loader, device)

    print("\n================ BEST VALIDATION ================")
    print("Best epoch                 :", best_epoch)
    print(f"Best validation node acc   : {best_valid_node_acc:.4f}")
    print(f"Best validation geometry acc: {best_valid_case_acc:.4f}")

    print("\n========== HELD-OUT CANTILEVER TEST =============")
    print(f"Node-level accuracy        : {test_result['node_accuracy']:.4f}")
    print(
        "Case-level strict accuracy : "
        f"{test_result['correct_cases']} / {test_result['total_cases']} "
        f"= {test_result['case_accuracy']:.4f}"
    )

    print("\nNode-level Confusion Matrix")
    print(test_result["confusion_matrix"])

    print("\nNode-level Classification Report")
    print(test_result["classification_report"])

    print("Saved best model:", MODEL_OUTPUT_FILE)

    incorrect_names = test_result["incorrect_case_names"]
    if incorrect_names:
        print("\nStrictly incorrect cantilever cases (first 30)")
        for name in incorrect_names[:30]:
            print(" -", name)


if __name__ == "__main__":
    main()
