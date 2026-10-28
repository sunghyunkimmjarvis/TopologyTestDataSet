# ============================================================
# 04_Check_Graph_Topology.py
#
# 목적
# ------------------------------------------------------------
# "왜 GNN 이 MLP 를 못 이겼는가" 를 학습을 돌리지 않고
# 데이터 자체만 보고 진단하는 스크립트.
#
# 아래 3가지를 계산합니다.
#
#  (1) 형상군 안에 실제로 서로 다른 그래프 위상이 몇 종류인가
#      -> 1종이면 인접행렬이 사실상 "상수"이고,
#         GNN 이 MLP 를 이길 수 있는 정보가 원천적으로 없습니다.
#
#  (2) Feature-only ceiling  (MLP 의 이론적 최대 정확도)
#      -> 노드 특징 벡터가 완전히 같은데 라벨이 다른 노드는
#         어떤 MLP 도 절대 구분할 수 없습니다.
#         "구분 가능한 만큼 다 맞혔을 때"의 정확도 상한.
#
#  (3) 1-WL ceiling  (message passing GNN 의 이론적 최대 정확도)
#      -> Weisfeiler-Lehman 색 재정의를 K번 반복한 뒤에도
#         색이 같은데 라벨이 다른 노드는 어떤 MP-GNN 도 구분 불가.
#         (Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019)
#
# 실행 결과의 읽는 법
# ------------------------------------------------------------
#   측정된 MLP 정확도  ==  (2) 값     -> MLP 는 이미 천장에 닿았다
#   측정된 GNN 정확도  <   (3) 값     -> 모델/학습 문제 (개선 여지 있음)
#   (2) == (3)                        -> 그래프 구조가 정보를 전혀 안 준다
#
# 사용법
# ------------------------------------------------------------
#   아래 CONFIG 의 EXCEL_FILES 와 USE_COORDINATE 만 맞춰서 실행.
#   pandas / numpy 만 있으면 되고 torch 는 필요 없습니다.
# ============================================================

import os
import csv
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================
# cantilever_EDGE_Feat / Bolt_EDGE_Feat / BookShelves_EDGE_Feat / Flange_EDGE_Feat / LshapedBracket_EDGE_Feat
# cantilever_noCoord_yesEdgeFeat / Bolt_noCoord_yesEdgeFeat / BookShelves_noCoord_yesEdgeFeat / Flange_noCoord_yesEdgeFeat / LshapedBracket_noCoord_yesEdgeFeat
# 좌표 포함 파일이면 True, *_without_Coordinate.xlsx 면 False
USE_COORDINATE = False

EXCEL_FILES = [
    r"TestDataResult_Excel\\cantilever_noCoord_yesEdgeFeat.xlsx",
    r"TestDataResult_Excel\\BookShelves_noCoord_yesEdgeFeat.xlsx",
    r"TestDataResult_Excel\\Bolt_noCoord_yesEdgeFeat.xlsx",
    r"TestDataResult_Excel\\LshapedBracket_noCoord_yesEdgeFeat.xlsx",
    r"TestDataResult_Excel\\Flange_noCoord_yesEdgeFeat.xlsx",
]

# WL 반복 횟수 (GNN 층 수와 맞추면 됩니다. 기존 모델은 2층)
WL_ITERATIONS = 2

# 부동소수 비교용 반올림 자릿수
ROUND_DECIMALS = 6

RESULT_CSV = "topology_check_result.csv"


if USE_COORDINATE:
    FEATURE_START_COL = 1
    FEATURE_END_COL = 13
    ADJ_START_COL = 14
else:
    FEATURE_START_COL = 1
    FEATURE_END_COL = 10
    ADJ_START_COL = 11


# ============================================================
# 2. Excel -> (특징, 라벨, 인접행렬) 블록 리스트
# ============================================================

def load_blocks(excel_file):

    df = pd.read_excel(excel_file, header=None)

    blocks = []
    current_rows = []

    for i in range(len(df)):

        row = df.iloc[i]

        if row.isna().all():
            if current_rows:
                blocks.append(pd.DataFrame(current_rows))
                current_rows = []
        else:
            current_rows.append(row)

    if current_rows:
        blocks.append(pd.DataFrame(current_rows))

    cases = []

    for b in blocks:

        b = b.reset_index(drop=True)

        feature_df = b.iloc[:, FEATURE_START_COL:FEATURE_END_COL]

        X = feature_df.iloc[:, :-1].values.astype(float)
        Y = feature_df.iloc[:, -1].values.astype(int)

        n = len(X)

        available = max(0, b.shape[1] - ADJ_START_COL)
        read_cols = min(n, available)

        raw = (
            b.iloc[:n, ADJ_START_COL:ADJ_START_COL + read_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=int)
        )

        A = np.zeros((n, n), dtype=int)
        A[:, :read_cols] = raw
        A = np.maximum(A, A.T)      # 상삼각만 있어도 대칭 복원
        np.fill_diagonal(A, 0)

        cases.append((X, Y, A))

    return cases


# ============================================================
# 3. WL 색 재정의
# ============================================================

def neighbors_of(A):
    return [np.nonzero(A[i])[0].tolist() for i in range(len(A))]


def wl_colors(A, init_colors, iterations):
    """
    init_colors : 각 노드의 초기 색 (hashable)
    iterations  : 반복 횟수 (GNN 층 수에 해당)
    반환        : 반복 후 각 노드의 색
    """

    nbrs = neighbors_of(A)
    colors = list(init_colors)

    for _ in range(iterations):

        new_colors = []

        for i in range(len(colors)):
            # 이웃 색은 순서에 무관하므로 정렬해서 multiset 으로 사용
            sig = (colors[i], tuple(sorted(colors[j] for j in nbrs[i])))
            new_colors.append(sig)

        colors = new_colors

    return colors


def structure_hash(A, iterations=3):
    """
    노드 특징을 무시하고 '위상만' 비교하기 위한 그래프 해시.
    초기 색을 모두 동일하게(차수 기반) 두고 WL 을 돌린 뒤
    색 multiset 을 정렬해 그래프 단위 지문을 만든다.
    """

    n = len(A)
    colors = [int(A[i].sum()) for i in range(n)]

    nbrs = neighbors_of(A)

    for _ in range(iterations):
        colors = [
            hash((colors[i], tuple(sorted(colors[j] for j in nbrs[i]))))
            for i in range(n)
        ]

    return hash((n, tuple(sorted(colors))))


# ============================================================
# 4. 상한 계산
# ============================================================

def ceiling_from_groups(group_to_label_counts):
    """
    같은 색(구분 불가능한 그룹) 안에서는 다수 라벨만 맞힐 수 있다.
    상한 = sum(그룹별 최빈 라벨 개수) / 전체 노드 수
    """

    total = 0
    best = 0

    for counter in group_to_label_counts.values():
        s = sum(counter.values())
        total += s
        best += max(counter.values())

    return best / total if total else float("nan")


def analyze_file(excel_file):

    print("=" * 74)
    print("FILE :", excel_file)
    print("=" * 74)

    cases = load_blocks(excel_file)

    n_cases = len(cases)
    n_nodes = sum(len(X) for X, _, _ in cases)

    print(f"  cases (solid bodies) : {n_cases}")
    print(f"  total nodes (faces)  : {n_nodes}")
    print(f"  nodes per case       : "
          f"min {min(len(X) for X, _, _ in cases)} / "
          f"max {max(len(X) for X, _, _ in cases)}")

    # ---- 라벨 분포 ----
    all_labels = np.concatenate([Y for _, Y, _ in cases])
    uniq, cnt = np.unique(all_labels, return_counts=True)
    print("  label distribution   : "
          + ", ".join(f"{u}:{c} ({c / n_nodes:.1%})" for u, c in zip(uniq, cnt)))

    majority = cnt.max() / n_nodes
    print(f"  majority baseline    : {majority:.4f}"
          f"   <- 전부 다수 클래스로 찍었을 때의 정확도")

    # ---- (1) 위상 종류 수 ----
    from collections import Counter, defaultdict

    topo = Counter(structure_hash(A) for _, _, A in cases)

    print()
    print(f"  [1] distinct graph topologies : {len(topo)}")

    if len(topo) == 1:
        print("      => 모든 케이스의 인접행렬이 동일합니다.")
        print("         인접행렬이 상수이므로 GNN 이 쓸 수 있는 추가 정보가 없습니다.")
    else:
        top = topo.most_common(5)
        print("      상위 위상별 케이스 수 :",
              ", ".join(str(c) for _, c in top))

    # ---- (2) feature-only ceiling (MLP 상한) ----
    feat_groups = defaultdict(Counter)

    for X, Y, _ in cases:
        for xr, yr in zip(X, Y):
            key = tuple(np.round(xr, ROUND_DECIMALS))
            feat_groups[key][int(yr)] += 1

    mlp_ceiling = ceiling_from_groups(feat_groups)

    print()
    print(f"  [2] feature-only ceiling (MLP 상한)  : {mlp_ceiling:.4f}")
    print(f"      구분 불가능한 특징 그룹 수        : {len(feat_groups)}")

    # ---- (3) 1-WL ceiling (message passing GNN 상한) ----
    wl_groups = defaultdict(Counter)

    for X, Y, A in cases:
        init = [tuple(np.round(xr, ROUND_DECIMALS)) for xr in X]
        cols = wl_colors(A, init, WL_ITERATIONS)
        for c, yr in zip(cols, Y):
            wl_groups[c][int(yr)] += 1

    gnn_ceiling = ceiling_from_groups(wl_groups)

    print(f"  [3] {WL_ITERATIONS}-round WL ceiling (GNN 상한) : {gnn_ceiling:.4f}")
    print(f"      구분 불가능한 WL 색 그룹 수       : {len(wl_groups)}")

    # ---- 해석 ----
    print()
    gain = gnn_ceiling - mlp_ceiling

    if abs(gain) < 1e-9:
        print("  [해석] GNN 상한 == MLP 상한")
        print("         그래프 구조를 써도 이론적으로 더 맞힐 수 있는 노드가 없습니다.")
        print("         => 모델을 바꿔도 소용없고, 표현(엣지 특징)이나")
        print("            데이터(위상 변화)를 바꿔야 합니다.")
    else:
        print(f"  [해석] 그래프 구조로 추가로 얻을 수 있는 여지 : +{gain:.4f}")
        print("         => GNN 이 이 값에 못 미친다면 모델/학습 쪽 개선 여지가 있습니다.")

    print()

    return {
        "file": os.path.basename(excel_file),
        "use_coordinate": USE_COORDINATE,
        "cases": n_cases,
        "nodes": n_nodes,
        "distinct_topologies": len(topo),
        "majority_baseline": round(float(majority), 6),
        "mlp_ceiling_feature_only": round(float(mlp_ceiling), 6),
        f"gnn_ceiling_wl{WL_ITERATIONS}": round(float(gnn_ceiling), 6),
        "structure_gain": round(float(gnn_ceiling - mlp_ceiling), 6),
    }


# ============================================================
# 5. MAIN
# ============================================================

def main():

    rows = []

    for f in EXCEL_FILES:

        if not os.path.exists(f):
            print(f"[skip] file not found: {f}")
            continue

        rows.append(analyze_file(f))

    if not rows:
        print("분석된 파일이 없습니다. EXCEL_FILES 경로를 확인하십시오.")
        return

    # ---- 최종 표 ----
    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)
    print(f"{'file':<38}{'cases':>7}{'topo':>6}{'major':>9}"
          f"{'MLP상한':>11}{'GNN상한':>11}{'gain':>9}")
    print("-" * 92)

    for r in rows:
        print(
            f"{r['file']:<38}"
            f"{r['cases']:>7}"
            f"{r['distinct_topologies']:>6}"
            f"{r['majority_baseline']:>9.4f}"
            f"{r['mlp_ceiling_feature_only']:>11.4f}"
            f"{r[f'gnn_ceiling_wl{WL_ITERATIONS}']:>11.4f}"
            f"{r['structure_gain']:>9.4f}"
        )

    print("-" * 92)
    print("  topo  : 형상군 안의 서로 다른 그래프 위상 개수 (1이면 인접행렬이 상수)")
    print("  major : 전부 다수 클래스로 찍었을 때의 정확도")
    print("  gain  : GNN 상한 - MLP 상한 = 그래프 구조로 얻을 수 있는 최대 이득")
    print()

    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", os.path.abspath(RESULT_CSV))


if __name__ == "__main__":
    main()
