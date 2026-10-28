# ============================================================
# 03-03_Model_Comparison_MultiSeed_GINE.py
#
# 공정한 Multi-seed 모델 비교
#   - MLP        : node feature만 사용
#   - GCN        : adjacency 사용
#   - GraphSAGE  : adjacency 사용
#   - GIN        : adjacency 사용
#   - GINE       : adjacency + Edge_Features 사용
#
# Excel workbook
#   첫 번째 sheet : 기존 Node Matrix + Label + Adjacency Matrix
#   Edge_Features : filename/source_node/target_node 및 edge attributes
#
# Edge attributes 기본 구성
#   convexity_code, dihedral_angle_deg, shared_edge_length,
#   edge_type_line, edge_type_arc, edge_type_spline
# ============================================================

import os
import sys
import csv
import copy
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GINEConv

from sklearn.metrics import confusion_matrix, f1_score


# ============================================================
# 1. CONFIG
# ============================================================

# cantilever_EDGE_Feat / Bolt_EDGE_Feat / BookShelves_EDGE_Feat / Flange_EDGE_Feat / LshapedBracket_EDGE_Feat
# cantilever_noCoord_yesEdgeFeat / Bolt_noCoord_yesEdgeFeat / BookShelves_noCoord_yesEdgeFeat / Flange_noCoord_yesEdgeFeat / LshapedBracket_noCoord_yesEdgeFeat
EXCEL_FILE = r"TestDataResult_Excel\\BookShelves_EDGE_Feat.xlsx"

# 좌표(avg_x, avg_y, avg_z) 를 쓰는 파일이면 True
# 좌표 제외(*_without_Coordinate.xlsx *_noCoord_yesEdgeFeat) 파일이면 False
USE_COORDINATE = True


TEST_EXCEL_FILE = None

# Edge feature sheet. 이름이 다르면 수정하십시오.
EDGE_SHEET_NAME = "Edge_Features"

# 실제 Edge_Features 헤더와 맞지 않는 이름은 아래 alias 함수에서 흡수합니다.
EDGE_FEATURE_COLUMNS = [
    "convexity_code",
    "dihedral_angle_deg",
    "shared_edge_length",
    "edge_type_line",
    "edge_type_arc",
    "edge_type_spline",
]

# True: adjacency에는 있는데 Edge_Features에 없는 edge가 있으면 즉시 중단
# False: 누락 edge에는 0-vector를 넣고 경고 출력
STRICT_EDGE_MATCH = True

# source_node/target_node가 1부터 시작하면 True, 0부터 시작하면 False.
# None이면 각 case별 node 값으로 자동 판정합니다.
EDGE_NODE_ONE_BASED = None

NUM_CLASSES = 3
CLASS_NAMES = ["general", "fixed", "load"]

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 200

TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2
SEED_LIST = [0, 1, 2, 3, 4]

MODEL_LIST = ["MLP", "GCN", "GraphSAGE", "GIN", "GINE"]

STANDARDIZE_NODE = True
STANDARDIZE_EDGE = True

RESULT_CSV = os.path.splitext(EXCEL_FILE)[0] + "_GINE_comparison.csv"


# ============================================================
# 2. NODE/ADJACENCY 열 위치
# ============================================================

if USE_COORDINATE:
    FEATURE_START_COL = 1
    FEATURE_END_COL = 13  # feature 11 + label 1
    ADJ_START_COL = 14
else:
    FEATURE_START_COL = 1
    FEATURE_END_COL = 10  # feature 8 + label 1
    ADJ_START_COL = 11


# ============================================================
# 3. LOG
# ============================================================

def setup_experiment_log(experiment_name="ModelComparison_GINE"):

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

    logger = logging.getLogger(f"logger_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    for text in [
        "=" * 70,
        "EXPERIMENT CONFIGURATION",
        "=" * 70,
        f"Run time          : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Python file       : {os.path.abspath(__file__)}",
        f"Excel file        : {EXCEL_FILE}",
        f"Test Excel file   : {TEST_EXCEL_FILE}",
        f"Edge sheet        : {EDGE_SHEET_NAME}",
        f"Edge features     : {EDGE_FEATURE_COLUMNS}",
        f"Strict edge match : {STRICT_EDGE_MATCH}",
        f"Use coordinate    : {USE_COORDINATE}",
        f"Standardize node  : {STANDARDIZE_NODE}",
        f"Standardize edge  : {STANDARDIZE_EDGE}",
        f"Models            : {MODEL_LIST}",
        f"Seeds             : {SEED_LIST}",
        f"Epochs            : {EPOCHS}",
        f"Batch size        : {BATCH_SIZE}",
        f"Learning rate     : {LEARNING_RATE}",
        f"Split ratio       : {TRAIN_RATIO}/{VALID_RATIO}/{TEST_RATIO}",
        "=" * 70,
    ]:
        logger.info(text)

    sys.stdout = TeeStream(sys.__stdout__, logger, logging.INFO)
    sys.stderr = TeeStream(sys.__stderr__, logger, logging.ERROR)
    print("Log file:", log_path)
    print()
    return log_path


# ============================================================
# 4. SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 완전 결정론이 필요하면 아래 두 줄을 활성화하십시오.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ============================================================
# 5. EXCEL -> PyG DATASET
# ============================================================

def normalize_name(value):
    """파일명 매칭용: 경로/확장자/대소문자/공백 차이를 제거."""
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("\\", "/")
    text = os.path.basename(text)
    text = os.path.splitext(text)[0]
    return "".join(text.lower().split())


def canonicalize_edge_columns(edge_df):
    """Edge_Features의 헤더 표기 차이를 canonical name으로 통일."""
    def key(col):
        return str(col).strip().lower().replace(" ", "_").replace("-", "_")

    aliases = {
        "filename": ["filename", "file_name", "file", "case", "case_name", "model_name"],
        "source_node": ["source_node", "source", "src", "source_index", "source_node_id"],
        "target_node": ["target_node", "target", "dst", "target_index", "target_node_id"],
        "convexity_code": ["convexity_code", "convexity", "signed_convexity"],
        "dihedral_angle_deg": ["dihedral_angle_deg", "dihedral_angle", "angle_deg", "dihedral"],
        "shared_edge_length": ["shared_edge_length", "edge_length", "length"],
        "edge_type_line": ["edge_type_line", "line", "is_line"],
        "edge_type_arc": ["edge_type_arc", "arc", "circle", "is_arc"],
        "edge_type_spline": ["edge_type_spline", "spline", "bspline", "is_spline"],
    }

    lookup = {key(c): c for c in edge_df.columns}
    rename = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if key(candidate) in lookup:
                rename[lookup[key(candidate)]] = canonical
                break

    edge_df = edge_df.rename(columns=rename).copy()

    required = ["filename", "source_node", "target_node"]
    missing_required = [c for c in required if c not in edge_df.columns]
    if missing_required:
        raise ValueError(
            f"Edge sheet required columns missing: {missing_required}. "
            f"Actual columns: {list(edge_df.columns)}"
        )

    # edge type이 문자열 1개 열로만 저장된 경우 one-hot 생성
    edge_type_col = None
    for c in edge_df.columns:
        if key(c) in {"edge_type", "curve_type", "edge_curve_type"}:
            edge_type_col = c
            break
    if edge_type_col is not None:
        type_text = edge_df[edge_type_col].astype(str).str.lower()
        if "edge_type_line" not in edge_df:
            edge_df["edge_type_line"] = type_text.str.contains("line").astype(float)
        if "edge_type_arc" not in edge_df:
            edge_df["edge_type_arc"] = type_text.str.contains("arc|circle").astype(float)
        if "edge_type_spline" not in edge_df:
            edge_df["edge_type_spline"] = type_text.str.contains("spline|bspline").astype(float)

    # 설정된 feature가 없으면 0으로 생성하되 경고
    for c in EDGE_FEATURE_COLUMNS:
        if c not in edge_df.columns:
            print(f"[WARNING] Edge feature column '{c}' missing -> filled with 0.")
            edge_df[c] = 0.0

    edge_df["_case_key"] = edge_df["filename"].map(normalize_name)
    edge_df["source_node"] = pd.to_numeric(edge_df["source_node"], errors="coerce")
    edge_df["target_node"] = pd.to_numeric(edge_df["target_node"], errors="coerce")
    edge_df = edge_df.dropna(subset=["source_node", "target_node"])
    edge_df["source_node"] = edge_df["source_node"].astype(int)
    edge_df["target_node"] = edge_df["target_node"].astype(int)

    for c in EDGE_FEATURE_COLUMNS:
        edge_df[c] = pd.to_numeric(edge_df[c], errors="coerce").fillna(0.0)

    return edge_df


def adjacency_to_edge_index(adj):
    src, dst = np.nonzero(adj)
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)


def infer_one_based(case_edges, num_nodes):
    if EDGE_NODE_ONE_BASED is not None:
        return EDGE_NODE_ONE_BASED
    if len(case_edges) == 0:
        return False
    values = pd.concat([case_edges["source_node"], case_edges["target_node"]])
    # 0이 존재하면 0-based. 최솟값 1이고 최댓값 <= num_nodes면 1-based.
    if (values == 0).any():
        return False
    return int(values.min()) >= 1 and int(values.max()) <= num_nodes


def build_edge_attr(case_name, edge_index, num_nodes, edge_df):
    """
    adjacency에서 생성된 directed edge_index 순서와 정확히 같은 순서로 edge_attr 생성.
    Edge_Features가 undirected 1행이어도 (u,v)/(v,u)에 동일 속성을 복제.
    """
    edge_dim = len(EDGE_FEATURE_COLUMNS)
    case_key = normalize_name(case_name)
    case_edges = edge_df[edge_df["_case_key"] == case_key].copy()

    if case_edges.empty:
        msg = f"No Edge_Features rows matched case '{case_name}' (key='{case_key}')."
        if STRICT_EDGE_MATCH:
            raise ValueError(msg)
        print("[WARNING]", msg, "All edge_attr set to zero.")
        return torch.zeros((edge_index.shape[1], edge_dim), dtype=torch.float)

    one_based = infer_one_based(case_edges, num_nodes)
    offset = 1 if one_based else 0

    # 동일 node pair에 여러 행이 있으면 평균. 일반적으로는 공유 edge 1개이므로 1행.
    pair_values = {}
    for _, row in case_edges.iterrows():
        u = int(row["source_node"]) - offset
        v = int(row["target_node"]) - offset

        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(
                f"Case '{case_name}': edge node index out of range: ({u}, {v}), "
                f"num_nodes={num_nodes}, one_based={one_based}"
            )

        vec = row[EDGE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        key = tuple(sorted((u, v)))
        pair_values.setdefault(key, []).append(vec)

    pair_mean = {k: np.mean(v, axis=0) for k, v in pair_values.items()}

    attrs = []
    missing = []
    adjacency_pairs = set()
    for u, v in edge_index.t().tolist():
        key = tuple(sorted((int(u), int(v))))
        adjacency_pairs.add(key)
        if key in pair_mean:
            attrs.append(pair_mean[key])
        else:
            missing.append(key)
            attrs.append(np.zeros(edge_dim, dtype=np.float32))

    extra = sorted(set(pair_mean) - adjacency_pairs)
    missing = sorted(set(missing))

    if missing:
        msg = f"Case '{case_name}': adjacency edges without edge features: {missing[:10]}"
        if STRICT_EDGE_MATCH:
            raise ValueError(msg)
        print("[WARNING]", msg, f"(total={len(missing)}; zero-filled)")

    if extra:
        # Edge sheet 연결이 Node Matrix adjacency와 불일치하는 문제를 노출
        print(
            f"[WARNING] Case '{case_name}': Edge_Features pairs absent from adjacency: "
            f"{extra[:10]} (total={len(extra)})"
        )

    return torch.tensor(np.asarray(attrs), dtype=torch.float)


def process_one_case(block_df, edge_df):
    block_df = block_df.reset_index(drop=True)

    # A열의 첫 유효값을 case name으로 사용
    names = block_df.iloc[:, 0].dropna()
    if names.empty:
        raise ValueError("Case block has no filename/case name in column A.")
    case_name = str(names.iloc[0]).strip()

    feature_df = block_df.iloc[:, FEATURE_START_COL:FEATURE_END_COL]
    x_df = feature_df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    y_df = pd.to_numeric(feature_df.iloc[:, -1], errors="coerce")

    if x_df.isna().any().any() or y_df.isna().any():
        raise ValueError(f"Case '{case_name}': non-numeric or missing node feature/label found.")

    x = torch.tensor(x_df.to_numpy(dtype=np.float32), dtype=torch.float)
    y = torch.tensor(y_df.to_numpy(dtype=np.int64), dtype=torch.long)
    num_nodes = len(x_df)

    available_adj_cols = max(0, block_df.shape[1] - ADJ_START_COL)
    read_adj_cols = min(num_nodes, available_adj_cols)
    adj_df = block_df.iloc[:num_nodes, ADJ_START_COL:ADJ_START_COL + read_adj_cols]
    raw_adj = adj_df.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=int)

    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    adj[:, :read_adj_cols] = raw_adj
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)

    edge_index = adjacency_to_edge_index(adj)
    edge_attr = build_edge_attr(case_name, edge_index, num_nodes, edge_df)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        case_name=case_name,
    )


def load_dataset_from_excel(excel_file):
    print("Loading:", excel_file)

    book = pd.ExcelFile(excel_file)
    if EDGE_SHEET_NAME not in book.sheet_names:
        raise ValueError(
            f"Sheet '{EDGE_SHEET_NAME}' not found in {excel_file}. "
            f"Available sheets: {book.sheet_names}"
        )

    node_sheet = book.sheet_names[0]
    if node_sheet == EDGE_SHEET_NAME and len(book.sheet_names) > 1:
        node_sheet = book.sheet_names[1]

    df = pd.read_excel(excel_file, sheet_name=node_sheet, header=None)
    edge_df_raw = pd.read_excel(excel_file, sheet_name=EDGE_SHEET_NAME)
    edge_df = canonicalize_edge_columns(edge_df_raw)

    print("  Node sheet:", node_sheet, "shape:", df.shape)
    print("  Edge sheet shape:", edge_df.shape)

    dataset = []
    current_rows = []

    for row_index in range(len(df)):
        row = df.iloc[row_index]
        if row.isna().all():
            if current_rows:
                dataset.append(process_one_case(pd.DataFrame(current_rows), edge_df))
                current_rows = []
        else:
            current_rows.append(row)

    if current_rows:
        dataset.append(process_one_case(pd.DataFrame(current_rows), edge_df))

    if not dataset:
        raise ValueError("No graph case was loaded from Excel.")

    edge_dims = {d.edge_attr.shape[1] for d in dataset}
    if len(edge_dims) != 1:
        raise ValueError(f"Inconsistent edge feature dimensions: {edge_dims}")

    print("  Dataset size (cases):", len(dataset))
    print("  Edge feature dimension:", next(iter(edge_dims)))
    return dataset


# ============================================================
# 6. SPLIT / STANDARDIZATION
# ============================================================

def split_dataset(dataset, train_ratio, valid_ratio, test_ratio):
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/valid/test ratios must sum to 1.0")

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    total = len(indices)
    train_size = int(total * train_ratio)
    valid_size = int(total * valid_ratio)

    tr = indices[:train_size]
    va = indices[train_size:train_size + valid_size]
    te = indices[train_size + valid_size:]

    return [dataset[i] for i in tr], [dataset[i] for i in va], [dataset[i] for i in te]


def standardize_dataset(train_set, other_sets):
    """Node와 edge 통계 모두 train set에만 fit하여 leakage를 방지."""
    if not train_set:
        raise ValueError("Empty train set.")

    node_mean = node_std = edge_mean = edge_std = None

    if STANDARDIZE_NODE:
        all_x = torch.cat([d.x for d in train_set], dim=0)
        node_mean = all_x.mean(dim=0, keepdim=True)
        node_std = all_x.std(dim=0, keepdim=True)
        node_std[node_std < 1e-8] = 1.0

    if STANDARDIZE_EDGE:
        nonempty = [d.edge_attr for d in train_set if d.edge_attr.numel() > 0]
        if nonempty:
            all_e = torch.cat(nonempty, dim=0)
            edge_mean = all_e.mean(dim=0, keepdim=True)
            edge_std = all_e.std(dim=0, keepdim=True)
            edge_std[torch.isnan(edge_std) | (edge_std < 1e-8)] = 1.0

    def apply(ds):
        out = []
        for d in ds:
            x = (d.x - node_mean) / node_std if node_mean is not None else d.x.clone()
            edge_attr = d.edge_attr.clone()
            if edge_mean is not None and edge_attr.numel() > 0:
                edge_attr = (edge_attr - edge_mean) / edge_std

            out.append(Data(
                x=x,
                edge_index=d.edge_index.clone(),
                edge_attr=edge_attr,
                y=d.y.clone(),
                case_name=d.case_name,
            ))
        return out

    return apply(train_set), [apply(ds) for ds in other_sets]


# ============================================================
# 7. MODELS
# ============================================================

class MLPBaseline(nn.Module):
    def __init__(self, num_features, num_classes, edge_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 32)
        self.fc2 = nn.Linear(32, 64)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, edge_index=None, edge_attr=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.linear(x)


class GCN(nn.Module):
    def __init__(self, num_features, num_classes, edge_dim=None):
        super().__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.linear(x)


class GraphSAGE(nn.Module):
    def __init__(self, num_features, num_classes, edge_dim=None):
        super().__init__()
        self.conv1 = SAGEConv(num_features, 32)
        self.conv2 = SAGEConv(32, 64)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.linear(x)


class GIN(nn.Module):
    def __init__(self, num_features, num_classes, edge_dim=None):
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(num_features, 32), nn.ReLU(), nn.Linear(32, 32))
        mlp2 = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64))
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.linear(x)


class GINE(nn.Module):
    """GIN + edge attributes. GINEConv 내부에서 edge feature를 메시지에 결합."""
    def __init__(self, num_features, num_classes, edge_dim):
        super().__init__()
        if edge_dim is None or edge_dim <= 0:
            raise ValueError("GINE requires edge_dim > 0")

        mlp1 = nn.Sequential(nn.Linear(num_features, 32), nn.ReLU(), nn.Linear(32, 32))
        mlp2 = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64))

        # edge_dim을 지정하면 PyG가 각 layer 입력 차원에 맞게 edge encoder를 둡니다.
        self.conv1 = GINEConv(mlp1, train_eps=True, edge_dim=edge_dim)
        self.conv2 = GINEConv(mlp2, train_eps=True, edge_dim=edge_dim)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            raise ValueError("GINE forward requires edge_attr")
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return self.linear(x)


def build_model(name, num_features, num_classes, edge_dim):
    table = {
        "MLP": MLPBaseline,
        "GCN": GCN,
        "GraphSAGE": GraphSAGE,
        "GIN": GIN,
        "GINE": GINE,
    }
    if name not in table:
        raise ValueError(f"Unknown model: {name}")
    return table[name](num_features, num_classes, edge_dim)


# ============================================================
# 8. TRAIN / EVALUATE
# ============================================================

def model_forward(model, batch):
    return model(batch.x, batch.edge_index, batch.edge_attr)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model_forward(model, batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        # node classification loss이므로 node 수로 가중 평균
        total_loss += float(loss.detach()) * batch.num_nodes
        total_nodes += batch.num_nodes

    return total_loss / max(1, total_nodes)


@torch.no_grad()
def evaluate(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    y_true_all, y_pred_all = [], []
    correct_cases = 0
    total_cases = 0

    for batch in loader:
        batch = batch.to(device)
        out = model_forward(model, batch)
        pred = out.argmax(dim=1)

        y_true_all.append(batch.y.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())

        for g in range(batch.num_graphs):
            mask = batch.batch == g
            correct_cases += int(bool((pred[mask] == batch.y[mask]).all()))
            total_cases += 1

    if not y_true_all:
        raise ValueError("Evaluation loader is empty.")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    node_acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(
        y_true, y_pred, average="macro",
        labels=list(range(num_classes)), zero_division=0
    ))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.diag(cm) / cm.sum(axis=1)
    recall = np.nan_to_num(recall)

    return {
        "node_acc": node_acc,
        "macro_f1": macro_f1,
        "case_acc": correct_cases / max(1, total_cases),
        "correct_cases": correct_cases,
        "total_cases": total_cases,
        "confusion_matrix": cm,
        "recall_per_class": recall,
    }


def run_one_setting(model_name, train_set, valid_set, test_set,
                    num_features, edge_dim, device, seed):
    set_seed(seed)

    model = build_model(model_name, num_features, NUM_CLASSES, edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # DataLoader shuffle까지 model별 동일하게 만들기 위한 generator
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    best_score = -1.0
    best_state = None
    best_epoch = -1
    history = []

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = evaluate(model, valid_loader, device)
        history.append((epoch, loss, val["node_acc"], val["macro_f1"], val["case_acc"]))

        if val["macro_f1"] > best_score:
            best_score = val["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"    [{model_name} seed{seed}] epoch {epoch:04d} | "
                f"loss {loss:.4f} | val node {val['node_acc']:.4f} | "
                f"val macroF1 {val['macro_f1']:.4f} | val geom {val['case_acc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_result = evaluate(model, test_loader, device)
    test_result["best_epoch"] = best_epoch
    test_result["history"] = history
    return test_result


# ============================================================
# 9. MAIN
# ============================================================

def main():
    setup_experiment_log()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print()

    full_dataset = load_dataset_from_excel(EXCEL_FILE)
    external_test = (
        load_dataset_from_excel(TEST_EXCEL_FILE)
        if TEST_EXCEL_FILE is not None else None
    )

    num_features = full_dataset[0].num_node_features
    edge_dim = full_dataset[0].edge_attr.shape[1]
    print("Number of node features:", num_features)
    print("Number of edge features:", edge_dim)
    print()

    results = {m: [] for m in MODEL_LIST}

    for seed in SEED_LIST:
        print("=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)
        set_seed(seed)

        if external_test is None:
            train_set, valid_set, test_set = split_dataset(
                full_dataset, TRAIN_RATIO, VALID_RATIO, TEST_RATIO
            )
        else:
            ratio_sum = TRAIN_RATIO + VALID_RATIO
            train_set, valid_set, _ = split_dataset(
                full_dataset,
                TRAIN_RATIO / ratio_sum,
                VALID_RATIO / ratio_sum,
                0.0,
            )
            test_set = external_test

        if STANDARDIZE_NODE or STANDARDIZE_EDGE:
            train_set, (valid_set, test_set) = standardize_dataset(
                train_set, [valid_set, test_set]
            )

        print(
            f"  train / valid / test cases : "
            f"{len(train_set)} / {len(valid_set)} / {len(test_set)}"
        )
        print()

        for model_name in MODEL_LIST:
            res = run_one_setting(
                model_name, train_set, valid_set, test_set,
                num_features, edge_dim, device, seed
            )
            results[model_name].append(res)

            print()
            print(
                f"  >> [{model_name} seed{seed}] TEST "
                f"node acc {res['node_acc']:.4f} | "
                f"macro-F1 {res['macro_f1']:.4f} | "
                f"geometry strict {res['correct_cases']}/{res['total_cases']} "
                f"= {res['case_acc']:.4f} | best epoch {res['best_epoch']}"
            )
            print("     recall per class:", {
                CLASS_NAMES[i]: round(float(v), 4)
                for i, v in enumerate(res["recall_per_class"])
            })
            print("     confusion matrix (test)")
            print("     " + str(res["confusion_matrix"]).replace("\n", "\n     "))
            print()

    print()
    print("=" * 86)
    print(
        f"SUMMARY ({EXCEL_FILE}, coordinate={USE_COORDINATE}, "
        f"edge_dim={edge_dim}, seeds={SEED_LIST})"
    )
    print("=" * 86)
    print(f"{'Model':<14}{'Node Acc':>20}{'Macro-F1':>20}{'Geometry Strict':>24}")
    print("-" * 86)

    rows = []
    for model_name in MODEL_LIST:
        node = np.array([r["node_acc"] for r in results[model_name]])
        f1 = np.array([r["macro_f1"] for r in results[model_name]])
        geom = np.array([r["case_acc"] for r in results[model_name]])

        print(
            f"{model_name:<14}"
            f"{node.mean():>11.4f} ±{node.std():.4f}"
            f"{f1.mean():>11.4f} ±{f1.std():.4f}"
            f"{geom.mean():>15.4f} ±{geom.std():.4f}"
        )

        rows.append({
            "excel_file": EXCEL_FILE,
            "use_coordinate": USE_COORDINATE,
            "edge_features": ",".join(EDGE_FEATURE_COLUMNS),
            "model": model_name,
            "n_seeds": len(SEED_LIST),
            "node_acc_mean": round(float(node.mean()), 6),
            "node_acc_std": round(float(node.std()), 6),
            "macro_f1_mean": round(float(f1.mean()), 6),
            "macro_f1_std": round(float(f1.std()), 6),
            "geometry_strict_mean": round(float(geom.mean()), 6),
            "geometry_strict_std": round(float(geom.std()), 6),
        })

    print("-" * 86)
    print()
    print("[Topology / edge contribution interpretation]")
    print(" - MLP vs GIN     : binary adjacency/topology의 추가 기여")
    print(" - GIN vs GINE    : edge semantics(이면각, convexity, 길이, type)의 추가 기여")
    print(" - MLP vs GINE    : 전체 relational representation의 순 기여")
    print(" - validation/test는 모든 모델에 동일한 split과 seed를 사용")
    print()

    os.makedirs(os.path.dirname(os.path.abspath(RESULT_CSV)), exist_ok=True)
    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", os.path.abspath(RESULT_CSV))


if __name__ == "__main__":
    main()
