# ============================================================
# 03-03_Model_Comparison_MultiSeed.py
#
# 목적
# ------------------------------------------------------------
# 기존 03 / 03-01 스크립트의 "GCN vs MLP" 비교를
# 아래 4개 모델 x 여러 seed 로 확장한 공정 비교 스크립트.
#
#   - MLP        : 그래프 구조를 전혀 쓰지 않는 baseline
#   - GCN        : 기존에 쓰던 모델 (GCNConv)
#   - GraphSAGE  : 자기 자신(root) 특징을 별도 가중치로 보존
#   - GIN        : (1+eps)*x_root + sum(neighbors), 1-WL 상한까지 표현 가능
#
# 기존 대비 보완한 점
# ------------------------------------------------------------
#   1) seed 를 여러 개 돌려 평균 +- 표준편차로 보고
#   2) validation 성능이 가장 좋은 시점의 모델을 저장해서 test
#      (기존에는 마지막 epoch 모델로 test 했음)
#   3) 입력 특징 표준화 (train set 에만 fit)
#   4) accuracy 뿐 아니라 macro-F1 / 클래스별 recall 도 보고
#      (클래스 불균형 때문에 accuracy 만 보면 과대평가됨)
#   5) 4개 모델이 완전히 동일한 데이터 분할/에폭/최적화기를 사용
#
# Excel format (기존과 동일)
# ------------------------------------------------------------
#   A열                : file name / case name
#   B열 ~ 빈 컬럼 전   : feature matrix + label (마지막 열이 label)
#   빈 컬럼 이후       : adjacency matrix
#   빈 행              : solid body(그래프 케이스) 구분자
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
from torch_geometric.nn import GCNConv, SAGEConv, GINConv

from sklearn.metrics import confusion_matrix, f1_score


# ============================================================
# 1. CONFIG  --  여기만 고치면 됩니다
# ============================================================

# cantilever / Bolt / BookShelves / Flange / LshapedBracket
# cantilever_without_Coordinate / Bolt_without_Coordinate / BookShelves_without_Coordinate / Flange_without_Coordinate / LshapedBracket_without_Coordinate
EXCEL_FILE = r"TestDataResult_Excel\\BookShelves_without_Coordinate.xlsx"

# 좌표(avg_x, avg_y, avg_z) 를 쓰는 파일이면 True
# 좌표 제외(*_without_Coordinate.xlsx) 파일이면 False
USE_COORDINATE = False

# 별도 test 용 Excel 이 있으면 경로를 넣고, 없으면 None
# (None 이면 EXCEL_FILE 을 train/valid/test 로 분할)
TEST_EXCEL_FILE = None

NUM_CLASSES = 3
CLASS_NAMES = ["general", "fixed", "load"]   # 실제 라벨 정의에 맞게 확인 필요

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 200

TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2

# 여러 번 반복해서 평균 +- 표준편차를 냅니다
SEED_LIST = [0, 1, 2, 3, 4]

# 비교할 모델
MODEL_LIST = ["MLP", "GCN", "GraphSAGE", "GIN"]

# 입력 특징 표준화 (권장: True)
STANDARDIZE = True

RESULT_CSV = EXCEL_FILE.split(".")[0] + ".csv"


# ============================================================
# 2. 열 위치 설정
# ============================================================

if USE_COORDINATE:
    # feature 11개 + label 1개 = 12열 (B ~ M)
    FEATURE_START_COL = 1
    FEATURE_END_COL = 13
    ADJ_START_COL = 14
else:
    # feature 8개 + label 1개 = 9열 (B ~ J)
    FEATURE_START_COL = 1
    FEATURE_END_COL = 10
    ADJ_START_COL = 11


# ============================================================
# 3. 로그
# ============================================================

def setup_experiment_log(experiment_name="ModelComparison"):

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

    logger.info("=" * 70)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Run time        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Python file     : {os.path.abspath(__file__)}")
    logger.info(f"Excel file      : {EXCEL_FILE}")
    logger.info(f"Test Excel file : {TEST_EXCEL_FILE}")
    logger.info(f"Use coordinate  : {USE_COORDINATE}")
    logger.info(f"Standardize     : {STANDARDIZE}")
    logger.info(f"Models          : {MODEL_LIST}")
    logger.info(f"Seeds           : {SEED_LIST}")
    logger.info(f"Epochs          : {EPOCHS}")
    logger.info(f"Batch size      : {BATCH_SIZE}")
    logger.info(f"Learning rate   : {LEARNING_RATE}")
    logger.info(f"Split ratio     : {TRAIN_RATIO}/{VALID_RATIO}/{TEST_RATIO}")
    logger.info("=" * 70)

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


# ============================================================
# 5. Excel -> PyG Dataset
# ============================================================

def adjacency_to_edge_index(adj):

    src, dst = np.nonzero(adj)

    edge_index = torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long
    )

    return edge_index


def process_one_case(block_df):

    block_df = block_df.reset_index(drop=True)

    feature_df = block_df.iloc[:, FEATURE_START_COL:FEATURE_END_COL]

    x_df = feature_df.iloc[:, :-1]
    y_df = feature_df.iloc[:, -1]

    x = torch.tensor(x_df.values.astype(np.float64), dtype=torch.float)
    y = torch.tensor(y_df.values.astype(np.int64), dtype=torch.long)

    num_nodes = len(x_df)

    available_adj_cols = max(0, block_df.shape[1] - ADJ_START_COL)
    read_adj_cols = min(num_nodes, available_adj_cols)

    adj_df = block_df.iloc[
        :num_nodes,
        ADJ_START_COL:ADJ_START_COL + read_adj_cols
    ]

    raw_adj = (
        adj_df.apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
    )

    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    adj[:, :read_adj_cols] = raw_adj

    # 상삼각만 저장되어 있어도 대칭 복원
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)

    return Data(
        x=x,
        edge_index=adjacency_to_edge_index(adj),
        y=y
    )


def load_dataset_from_excel(excel_file):

    print("Loading:", excel_file)

    df = pd.read_excel(excel_file, header=None)
    print("  Excel shape:", df.shape)

    dataset = []
    current_rows = []

    for row_index in range(len(df)):

        row = df.iloc[row_index]

        if row.isna().all():
            if len(current_rows) > 0:
                dataset.append(process_one_case(pd.DataFrame(current_rows)))
                current_rows = []
        else:
            current_rows.append(row)

    if len(current_rows) > 0:
        dataset.append(process_one_case(pd.DataFrame(current_rows)))

    print("  Dataset size (cases):", len(dataset))

    if len(dataset) == 0:
        raise ValueError("No graph case was loaded from Excel.")

    return dataset


# ============================================================
# 6. Split / 표준화
# ============================================================

def split_dataset(dataset, train_ratio, valid_ratio, test_ratio):

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    total = len(indices)
    train_size = int(total * train_ratio)
    valid_size = int(total * valid_ratio)

    tr = indices[:train_size]
    va = indices[train_size:train_size + valid_size]
    te = indices[train_size + valid_size:]

    return (
        [dataset[i] for i in tr],
        [dataset[i] for i in va],
        [dataset[i] for i in te],
    )


def standardize(train_set, other_sets):
    """
    train set 의 노드 특징으로만 평균/표준편차를 계산하고
    (data leakage 방지) 나머지 set 에 동일하게 적용한다.
    """

    all_x = torch.cat([d.x for d in train_set], dim=0)

    mean = all_x.mean(dim=0, keepdim=True)
    std = all_x.std(dim=0, keepdim=True)

    # 상수 컬럼(std=0) 방어
    std[std < 1e-8] = 1.0

    def apply(ds):
        out = []
        for d in ds:
            nd = Data(
                x=(d.x - mean) / std,
                edge_index=d.edge_index,
                y=d.y
            )
            out.append(nd)
        return out

    return apply(train_set), [apply(s) for s in other_sets]


# ============================================================
# 7. MODELS
#    4개 모델 모두 은닉 폭을 32 -> 64 -> num_classes 로 통일
#    (기존 스크립트의 GCN/MLP 구조와 동일한 용량)
# ============================================================

class MLPBaseline(nn.Module):
    """그래프 구조를 전혀 쓰지 않는 baseline (edge_index 무시)"""

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 32)
        self.fc2 = nn.Linear(32, 64)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.linear(x)


class GCN(nn.Module):
    """기존에 쓰던 모델. 자기 자신과 이웃을 정규화 평균으로 섞는다."""

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.linear(x)


class GraphSAGE(nn.Module):
    """root 특징과 이웃 집계를 서로 다른 가중치로 처리 -> MLP 이하로 내려가지 않음"""

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(num_features, 32)
        self.conv2 = SAGEConv(32, 64)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.linear(x)


class GIN(nn.Module):
    """(1+eps)*x_root + sum(neighbors). 1-WL 상한까지 표현 가능한 모델"""

    def __init__(self, num_features, num_classes):
        super().__init__()

        mlp1 = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        mlp2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.conv1 = GINConv(mlp1, train_eps=True)
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.linear(x)


def build_model(name, num_features, num_classes):

    table = {
        "MLP": MLPBaseline,
        "GCN": GCN,
        "GraphSAGE": GraphSAGE,
        "GIN": GIN,
    }

    if name not in table:
        raise ValueError(f"Unknown model: {name}")

    return table[name](num_features, num_classes)


# ============================================================
# 8. Train / Evaluate
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.num_graphs

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, device, num_classes=NUM_CLASSES):
    """
    노드 단위 정확도 / macro-F1 / 혼동행렬 과
    형상(케이스) 단위 strict accuracy 를 한 번에 계산.
    """

    model.eval()

    y_true_all = []
    y_pred_all = []

    correct_cases = 0
    total_cases = 0

    for batch in loader:
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)

        y_true_all.append(batch.y.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())

        # 케이스 단위: 한 그래프의 모든 노드가 맞아야 정답
        graph_id = batch.batch
        for g in range(batch.num_graphs):
            mask = (graph_id == g)
            if bool((pred[mask] == batch.y[mask]).all()):
                correct_cases += 1
            total_cases += 1

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    node_acc = float((y_true == y_pred).mean())

    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro",
                 labels=list(range(num_classes)), zero_division=0)
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.diag(cm) / cm.sum(axis=1)
    recall = np.nan_to_num(recall)

    case_acc = correct_cases / max(1, total_cases)

    return {
        "node_acc": node_acc,
        "macro_f1": macro_f1,
        "case_acc": case_acc,
        "correct_cases": correct_cases,
        "total_cases": total_cases,
        "confusion_matrix": cm,
        "recall_per_class": recall,
    }


def run_one_setting(model_name, train_set, valid_set, test_set,
                    num_features, device, seed):
    """
    한 (모델, seed) 조합을 학습하고,
    validation macro-F1 이 가장 좋았던 시점의 가중치로 test 를 평가한다.
    """

    set_seed(seed)

    model = build_model(model_name, num_features, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
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

        # ---- 최고 성능 시점 저장 (기존 스크립트에 없던 부분) ----
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

    # ---- best checkpoint 복원 후 test ----
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

    if TEST_EXCEL_FILE is not None:
        external_test = load_dataset_from_excel(TEST_EXCEL_FILE)
    else:
        external_test = None

    num_features = full_dataset[0].num_node_features
    print("Number of node features:", num_features)
    print()

    # 결과 저장용:  results[model_name] = list of dict (seed 별)
    results = {m: [] for m in MODEL_LIST}

    for seed in SEED_LIST:

        print("=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)

        # 분할은 seed 마다 새로 하되, 그 seed 안에서는 4개 모델이 동일 분할 사용
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
                0.0
            )
            test_set = external_test

        if STANDARDIZE:
            train_set, (valid_set, test_set) = standardize(
                train_set, [valid_set, test_set]
            )

        print(f"  train / valid / test cases : "
              f"{len(train_set)} / {len(valid_set)} / {len(test_set)}")
        print()

        for model_name in MODEL_LIST:

            res = run_one_setting(
                model_name, train_set, valid_set, test_set,
                num_features, device, seed
            )

            results[model_name].append(res)

            print()
            print(f"  >> [{model_name} seed{seed}] TEST "
                  f"node acc {res['node_acc']:.4f} | "
                  f"macro-F1 {res['macro_f1']:.4f} | "
                  f"geometry strict {res['correct_cases']}/{res['total_cases']} "
                  f"= {res['case_acc']:.4f} | best epoch {res['best_epoch']}")
            print("     confusion matrix (test)")
            print("     " + str(res["confusion_matrix"]).replace("\n", "\n     "))
            print()

    # --------------------------------------------------------
    # 최종 요약
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(f"SUMMARY  ({EXCEL_FILE},  coordinate={USE_COORDINATE},  "
          f"seeds={SEED_LIST})")
    print("=" * 78)
    print(f"{'Model':<12}{'Node Acc':>18}{'Macro-F1':>18}{'Geometry Strict':>22}")
    print("-" * 78)

    rows = []

    for model_name in MODEL_LIST:

        node = np.array([r["node_acc"] for r in results[model_name]])
        f1 = np.array([r["macro_f1"] for r in results[model_name]])
        geom = np.array([r["case_acc"] for r in results[model_name]])

        print(
            f"{model_name:<12}"
            f"{node.mean():>10.4f} ±{node.std():.4f}"
            f"{f1.mean():>10.4f} ±{f1.std():.4f}"
            f"{geom.mean():>14.4f} ±{geom.std():.4f}"
        )

        rows.append({
            "excel_file": EXCEL_FILE,
            "use_coordinate": USE_COORDINATE,
            "model": model_name,
            "n_seeds": len(SEED_LIST),
            "node_acc_mean": round(float(node.mean()), 6),
            "node_acc_std": round(float(node.std()), 6),
            "macro_f1_mean": round(float(f1.mean()), 6),
            "macro_f1_std": round(float(f1.std()), 6),
            "geometry_strict_mean": round(float(geom.mean()), 6),
            "geometry_strict_std": round(float(geom.std()), 6),
        })

    print("-" * 78)
    print()
    print("[해석 가이드]")
    print(" - GIN / GraphSAGE 가 MLP 와 같은 값이면")
    print("   => 현재 그래프 표현에 노드 특징 이상의 정보가 없다는 뜻입니다.")
    print("      (모델 문제가 아니라 데이터/표현 설계의 문제)")
    print(" - GCN 만 MLP 보다 낮으면")
    print("   => GCN 의 정규화 평균 집계가 특징을 희석시킨 것입니다 (over-smoothing).")
    print(" - GIN / GraphSAGE 가 MLP 보다 확실히 높으면")
    print("   => 그래프 구조가 실제로 기여한다는 근거가 됩니다.")
    print(" - 04_Check_Graph_Topology.py 로 계산한 이론적 상한과 비교해 보십시오.")
    print()

    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", os.path.abspath(RESULT_CSV))


if __name__ == "__main__":
    main()
