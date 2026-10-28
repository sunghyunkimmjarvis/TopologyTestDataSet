# STEP B-Rep Graph Dataset and Boundary-Condition Inference

**Technical README — code-specific documentation of the ten supplied Python scripts**  
**Documentation date:** 2026-10-28  
**Scope:** dataset generation, Excel data format, face/edge representation, supervised boundary-condition inference, coordinate ablation, leave-one-family-out (LOFO), multi-seed comparison, and topology diagnostics.

> **Important:** This README describes the uploaded source code as supplied, not an independently validated model performance claim. Example names and outputs are illustrative. Engineering boundary-condition labels must be verified against the intended loading and support scenario before using them as ground truth.

## Table of contents

1. [Research objective and pipeline](#1-research-objective-and-pipeline)
2. [Inventory of all ten scripts](#2-inventory-of-all-ten-scripts)
3. [Installation, execution environment and directory layout](#3-installation-execution-environment-and-directory-layout)
4. [STEP B-Rep parsing and graph construction](#4-step-b-rep-parsing-and-graph-construction)
5. [Excel output: ordinary face-feature dataset](#5-excel-output-ordinary-face-feature-dataset)
6. [Excel output: edge-feature-augmented dataset](#6-excel-output-edge-feature-augmented-dataset)
7. [Data preparation and PyTorch Geometric representation](#7-data-preparation-and-pytorch-geometric-representation)
8. [Detailed script-by-script documentation](#8-detailed-script-by-script-documentation)
9. [Experiments and evaluation metrics](#9-experiments-and-evaluation-metrics)
10. [Dataset combinations and ablation workflows](#10-dataset-combinations-and-ablation-workflows)
11. [Output artifacts and how to interpret them](#11-output-artifacts-and-how-to-interpret-them)
12. [Validation, reproducibility and limitations](#12-validation-reproducibility-and-limitations)
13. [Troubleshooting](#13-troubleshooting)
14. [Quick-start commands](#14-quick-start-commands)
15. [Suggested methods description and citation notes](#15-suggested-methods-description-and-citation-notes)

---

## 1. Research objective and pipeline

The scripts investigate **face-level inference of finite-element-analysis (FEA) boundary-condition categories** from CAD geometry represented as ISO STEP Boundary Representation (B-Rep). In the graph formulation, each topological face is a node, and two nodes are adjacent if their faces reference a common `EDGE_CURVE`. One target label is predicted per face: `0 = general/normal`, `1 = fixed`, `2 = load`.

```text
Parametric CAD design cases
         |
         v
ASCII STEP (.step / .stp, depending on generator)
         |
         v
STEP entity dictionary and ADVANCED_FACE extraction
         |
         +--> Per-face geometric/topological feature vectors
         |       (11 attributes; or 8 after removing XYZ coordinates)
         |
         +--> Shared EDGE_CURVE identifiers --> adjacency matrix
         |                                   |
         |                                   +--> Optional shared-edge geometry
         |                                        (angle, convexity, length, type)
         v
Aggregated Excel workbook in TestDataResult_Excel/
         |
         +--> Standard node-only workbook (.xlsx)
         |       [face features | labels | adjacency]
         |
         +--> Edge-feature workbook (*_EDGE_Feat.xlsx)
                 Node_Matrix + Edge_Features worksheets
         |
         v
PyTorch Geometric Data(x, edge_index, y[, edge_attr])
         |
         +--> Within-family MLP / GCN / GINE experiments
         +--> Without-coordinate ablation
         +--> Leave-one-family-out (LOFO) testing
         +--> Five-seed MLP / GCN / GraphSAGE / GIN / GINE comparison
         +--> Graph topology and feature/WL discriminability diagnostics
         |
         v
Console output, logs, CSV summaries, and some .pt checkpoints
```

**Unit of splitting:** graph/STEP case rather than an individual face. Faces from one case must not be divided across train and test; otherwise information from the same geometry can leak across partitions.

**Five family names used by the scripts:** `cantilever`, `Bolt`, `BookShelves`, `Flange`, `LshapedBracket`. The casing matters on case-sensitive file systems.

## 2. Inventory of all ten scripts

| Script | Input | Primary operation | Expected artifact |
|---|---|---|---|
| `01_FromSTEP_toExcel_FeatMatrixGenerated.py` | `TestData/<Target>/` STEP cases | Extract 11 node features, inferred label, face adjacency | `TestDataResult_Excel/<Target>.xlsx` (single default sheet) |
| `01_FromSTEP_toExcel_FeatMatrixGenerated_withEDGEFeature.py` | STEP cases | Same node representation plus shared-edge descriptors | `TestDataResult_Excel/<Target>_EDGE_Feat.xlsx` (`Node_Matrix`, `Edge_Features`) |
| `02_Train_DesignFamily.py` | Node-only workbook | Within-family GCN versus MLP | Console results, timestamped `.log` |
| `02_Train_DesignFamily_withEDGEFeature.py` | Two-sheet edge workbook | Within-family GINE versus MLP | Console, `.log`, `best_gine_edge_model.pt` |
| `03-01_Train_Feature_Adjacent_Matrix_without_XYZCoordinate.py` | Manually prepared 8-feature workbook | Coordinate ablation: GCN versus MLP | Console results, `.log` |
| `03-02_LOFO_Train4_TestCantilever.py` | Seen-family workbook(s) + held-out workbook(s) | GCN LOFO generalization | Console/evaluation and `gcn_lofo_test_cantilever.pt` |
| `03-02_LOFO_GINE_Edge_Train4_TestCantilever.py` | Two-sheet seen/held-out edge workbooks | GINE and MLP LOFO | Console/evaluation and `gine_lofo_test_cantilever_edge.pt` |
| `03-03_Model_Comparison_MultiSeed.py` | Node-only workbook | Five-seed comparison of MLP, GCN, GraphSAGE, GIN | `<EXCEL_FILE without extension>.csv`, `.log` |
| `03-03_Model_Comparison_MultiSeed_GINE.py` | Two-sheet edge workbook | Five-seed comparison of MLP, GCN, GraphSAGE, GIN, GINE | `<EXCEL_FILE without extension>_GINE_comparison.csv`, `.log` |
| `04_Check_Graph_Topology.py` | Node-style Excel workbooks specified in `EXCEL_FILES` | Topology diversity, majority baseline, feature-only/WL grouping scores | `topology_check_result.csv` |

The `03-*` numbering reflects experiments rather than a required sequential processing step. Generated workbooks must already exist before training and diagnostics.

## 3. Installation, execution environment and directory layout

### 3.1 Requirements

The code imports Python standard-library modules plus these third-party packages:

```bash
python -m pip install numpy pandas openpyxl scikit-learn torch torch-geometric
```

`numpy`, `pandas` and `openpyxl` suffice for STEP-to-Excel generation. The topology diagnostic additionally uses standard-library `csv` and does **not** require PyTorch. Training requires `torch`, `torch-geometric` and `scikit-learn`. Configure the PyTorch/CUDA combination for the target hardware using the official compatibility instructions. Exact tested package versions, operating system build and GPU model are **not recorded in these scripts**, so this README does not invent them.

### 3.2 Suggested directory arrangement

```text
project_root/
├── README.md
├── 01_FromSTEP_toExcel_FeatMatrixGenerated.py
├── 01_FromSTEP_toExcel_FeatMatrixGenerated_withEDGEFeature.py
├── 02_Train_DesignFamily.py
├── 02_Train_DesignFamily_withEDGEFeature.py
├── 03-01_Train_Feature_Adjacent_Matrix_without_XYZCoordinate.py
├── 03-02_LOFO_Train4_TestCantilever.py
├── 03-02_LOFO_GINE_Edge_Train4_TestCantilever.py
├── 03-03_Model_Comparison_MultiSeed.py
├── 03-03_Model_Comparison_MultiSeed_GINE.py
├── 04_Check_Graph_Topology.py
├── TestData/
│   ├── cantilever/
│   │   ├── cantilever_0001.step
│   │   └── ...
│   ├── Bolt/
│   ├── BookShelves/
│   ├── Flange/
│   └── LshapedBracket/
├── TestDataResult_Excel/
│   ├── cantilever.xlsx
│   ├── Flange_EDGE_Feat.xlsx
│   ├── *_without_Coordinate.xlsx      # derived separately, not generated by the two 01 scripts
│   └── additional combined workbooks   # assembled separately
├── logs/                             # written by many training scripts
├── topology_check_result.csv         # written by topology diagnostic
└── *.pt                              # checkpoints for specific training scripts
```

Examples are not a claim that those files are included in the upload. Relative paths in many scripts are interpreted from the **current working directory**, so run them from the project root where possible. The edge generator has a `--base-dir` option; the ordinary generator uses the relative directory configuration at the top of the file.

### 3.3 Five family names

| Family | `Target` value | Example ordinary workbook | Example edge workbook |
|---|---|---|---|
| Cantilever | `cantilever` | `cantilever.xlsx` | `cantilever_EDGE_Feat.xlsx` |
| Bolt | `Bolt` | `Bolt.xlsx` | `Bolt_EDGE_Feat.xlsx` |
| Bookshelf | `BookShelves` | `BookShelves.xlsx` | `BookShelves_EDGE_Feat.xlsx` |
| Flange | `Flange` | `Flange.xlsx` | `Flange_EDGE_Feat.xlsx` |
| L-shaped bracket | `LshapedBracket` | `LshapedBracket.xlsx` | `LshapedBracket_EDGE_Feat.xlsx` |

## 4. STEP B-Rep parsing and graph construction

### 4.1 Entity parsing

The generator reads STEP text with `encoding="utf-8", errors="ignore"`, builds a dictionary with a regular expression matching entity definitions `#id = ...;`, then collects records beginning with `ADVANCED_FACE`. References are traversed through STEP topology entities. The conceptual hierarchy is:

```text
MANIFOLD_SOLID_BREP / CLOSED_SHELL
     └── ADVANCED_FACE
           └── FACE_BOUND / FACE_OUTER_BOUND
                 └── EDGE_LOOP
                       └── ORIENTED_EDGE
                             └── EDGE_CURVE
                                   ├── VERTEX_POINT -> CARTESIAN_POINT
                                   └── LINE / CIRCLE / B_SPLINE_CURVE
```

The parsing implementation is **regular-expression and reference traversal based**, not a complete STEP standards-compliant parser. Its entity-definition pattern processes the assumed one-line form and may not correctly handle multiline entity records or all valid STEP constructs. STEP serialization conventions, exporter behavior and face order can affect extraction.

### 4.2 Node definition

A node corresponds to one `ADVANCED_FACE`. The code assigns a zero-based node index in the order faces are encountered in the entity dictionary; it does not add this index as a dedicated column to `Node_Matrix`. For example, the first face in a case maps to row 0 internally, the second to row 1, etc. Node-index order must be preserved in the companion adjacency and `Edge_Features` sheets.

### 4.3 Adjacency definition

Two faces `i` and `j` are adjacent when their extracted `EDGE_CURVE` ID sets intersect:

```python
shared_edges = face_edges[face_i_id] & face_edges[face_j_id]
adjacency[i, j] = adjacency[j, i] = int(bool(shared_edges))
```

The matrix is symmetric, binary, and has no diagonal self-loops. **Adjacency is topological contact via a shared B-Rep edge, not proximity of face centroids or a finite-element mesh-neighborhood relation.** One face pair may share multiple B-Rep edges even though the adjacency matrix records only one binary connection.

### 4.4 Feature extraction and engineering caveats

For each face, edge geometry categories are determined by looking at referenced curve entities. `CIRCLE` is encoded as ARC, `LINE` as LINE, and `B_SPLINE_CURVE` as SPLINE. Edge-length statistics in the node feature vector use the **endpoint-to-endpoint straight-line chord**, not exact curve arc length. Positive chord lengths are included; zero-length chords (e.g., a full circular edge represented with coincident endpoints) do not contribute to max/min/mean chord statistics.

Average X/Y/Z values are means over **unique extracted vertex coordinates**, plus a circle center when the face contains an edge categorized as ARC and a center can be obtained. These are not area-weighted face centroids. One face can therefore have coordinates that do not match a CAD kernel's face center of mass.

### 4.5 Label assignment

The 01 scripts initialize labels to zero and apply an example heuristic:

1. `fixed_face_idx = idxmin(avg_x)` is assigned **1 (fixed)**.
2. `load_face_idx = index of max(avg_z)` is assigned **2 (load)**; if this is the same face as the fixed face, the next highest-Z candidate is taken.
3. All other faces remain **0 (general/normal)**.

This is not general inference of physical support/loading semantics from geometry. It generates *candidate/synthetic labels under an assumed orientation*, and the face selected on a tied minimum/maximum can depend on face enumeration. **Review and manually correct the label column against the intended FEA load case before training.** In particular, separate pressure-loading surfaces, multiple support faces, assembly constraints and non-axis-aligned loads are outside the heuristic.

## 5. Excel output: ordinary face-feature dataset

### 5.1 Generator and output naming

Run `01_FromSTEP_toExcel_FeatMatrixGenerated.py` after setting `Target`. It traverses STEP case files in `TestData/<Target>/`, excludes the exact family-named placeholder `<Target>.step`, allows `.step` or `.stp` extensions, and writes:

```text
TestDataResult_Excel/<Target>.xlsx
```

The resulting workbook has **one default Excel sheet, no header row, and a blank row separating graph cases**. Each face contributes one row.

### 5.2 Exact column map (with XYZ coordinates)

| Excel column | Zero-based `iloc` | Content | Interpretation |
|---|---:|---|---|
| A | 0 | `filename` | STEP file name repeated for every face in the case |
| B | 1 | `total_edge_count` | Number of distinct extracted B-Rep edge IDs on that face |
| C | 2 | `line_edge_count` | Number of edges classified LINE |
| D | 3 | `arc_edge_count` | Number of edges classified ARC/CIRCLE |
| E | 4 | `spline_edge_count` | Number classified B_SPLINE_CURVE |
| F | 5 | `max_edge_length` | Maximum **positive endpoint chord length** |
| G | 6 | `min_edge_length` | Minimum **positive endpoint chord length** |
| H | 7 | `mean_edge_length` | Mean of positive endpoint chord lengths |
| I | 8 | `avg_x` | Mean unique extracted X coordinate |
| J | 9 | `avg_y` | Mean unique extracted Y coordinate |
| K | 10 | `avg_z` | Mean unique extracted Z coordinate |
| L | 11 | `surface_type` | Integer categorical encoding, **not one-hot** |
| M | 12 | `label` | Boundary-condition class 0, 1, or 2 |
| N | 13 | blank separator | Entire column intentionally empty |
| O onward | 14 onward | adjacency | Binary face-to-face adjacency, `n` columns for an `n`-face case |

Thus, `B:L` comprises **11 input features**, `M` is the **target label**, and `O` is the **first adjacency column**. In pandas, the code uses `FEATURE_START_COL = 1`, `FEATURE_END_COL = 13` (**exclusive**), `ADJ_START_COL = 14`. The feature-and-label slice is `df.iloc[:, 1:13]`, giving 12 columns, and the input `x` is the first 11 of those columns.

**Surface encoding (`surface_type`):**

| Value | Geometry |
|---:|---|
| 0 | PLANE |
| 1 | CYLINDRICAL_SURFACE |
| 2 | SPHERICAL_SURFACE |
| 3 | CONICAL_SURFACE |
| 4 | TOROIDAL_SURFACE |
| 5 | B_SPLINE_SURFACE / NURBS |
| 6 | Other / ETC |
| 7 | Fallback if a surface type is unexpectedly absent from the mapping |

Despite a comment saying "Surface One-Hot Encoding", the implementation writes **one integer category per face**. Models ingest it numerically unless preprocessing is changed; it is **not** a seven-column one-hot vector.

### 5.3 Case-block example

**Illustrative miniature example** (not a complete physical case):

```text
A (filename)   B ... L (11 features)      M(label)   N(blank)   O P Q ...
beam_001.step   [features for face 0]         1       [blank]    0 1 0 ...
beam_001.step   [features for face 1]         0       [blank]    1 0 1 ...
beam_001.step   [features for face 2]         2       [blank]    0 1 0 ...
[completely empty row separates cases]
beam_002.step   [features for face 0]         0       [blank]    0 1 ...
...
```

If a case has `n=6` faces, that case has six data rows, six adjacency columns `O:T`, and a logically 6×6 adjacency matrix. The next case follows after a fully blank separator row. When cases have different face counts, shorter cases have empty cells beyond their own adjacency width. In the node-only generator those blocks are padded to the maximum output width. **Never treat the padded empty columns as extra faces.**

### 5.4 Graph-level interpretation

For one case with `n` faces, the input and target dimensions are:

- Node feature matrix `X`: `n × 11`.
- Node label vector `y`: `n` integer entries.
- Binary adjacency matrix `A`: `n × n`.
- Physical graph: one CAD body/case, **not** one row in the spreadsheet.

The file contains geometry-derived features and labels, **not finite-element mesh elements, stresses, forces, material properties, solver loads, or boundary-condition magnitudes**.

## 6. Excel output: edge-feature-augmented dataset

### 6.1 Generator and execution

The edge generator supports command-line arguments:

```bash
python 01_FromSTEP_toExcel_FeatMatrixGenerated_withEDGEFeature.py --target Flange --base-dir .
```

Default `Target` in the uploaded script is `Flange`. The generator reads `.step` files (not `.stp` in its current file filter), skips `<Target>.step`, and produces:

```text
TestDataResult_Excel/<Target>_EDGE_Feat.xlsx
```

The workbook contains **two named worksheets**:

| Sheet | Header? | Granularity | Purpose |
|---|---|---|---|
| `Node_Matrix` | No | One row per face plus blank case separators | The same A–N/O+ node and adjacency layout as the ordinary generator |
| `Edge_Features` | Yes | One row per shared B-Rep `EDGE_CURVE` for a face pair | Relational geometry used to construct GINE edge attributes |

### 6.2 `Node_Matrix`

This sheet retains **A = filename, B:L = 11 features, M = label, N = blank, O onward = adjacency**, with a fully blank row between graph cases. Read it with:

```python
node_df = pd.read_excel(path, sheet_name="Node_Matrix", header=None)
```

Do **not** use the default first-row-as-header setting, as the first row represents a face rather than column names.

### 6.3 `Edge_Features`: all output columns

The generator sets the following ordered columns, preceded by `filename`. Unlike the node sheet, this sheet has a real header row and **does not use blank case-separator rows**.

| Excel column | Header | Meaning |
|---|---|---|
| A | `filename` | STEP case to which this shared edge belongs |
| B | `edge_curve_id` | STEP entity ID of the shared `EDGE_CURVE` (e.g., `#123`) |
| C | `source_face` | STEP `ADVANCED_FACE` entity ID of first face |
| D | `target_face` | STEP `ADVANCED_FACE` entity ID of second face |
| E | `source_node` | Zero-based index of first graph face |
| F | `target_node` | Zero-based index of second graph face |
| G | `convex_concave` | String category: `CONVEX`, `CONCAVE`, `SMOOTH`, `UNKNOWN` |
| H | `convexity_code` | `CONCAVE=-1`, `SMOOTH=0`, `CONVEX=1`, `UNKNOWN=0` |
| I | `dihedral_angle_deg` | Angle between approximated oriented face normals, in degrees |
| J | `shared_edge_length` | Calculated shared-edge length or approximation |
| K | `edge_type` | `LINE`, `ARC`, `SPLINE`, or `OTHER` |
| L | `edge_type_line` | 1 for LINE, otherwise 0 |
| M | `edge_type_arc` | 1 for ARC/CIRCLE, otherwise 0 |
| N | `edge_type_spline` | 1 for B-spline, otherwise 0 |
| O | `signed_convexity` | Triple-product value used in convexity classification, possibly NaN |

Read it with:

```python
edges = pd.read_excel(path, sheet_name="Edge_Features", header=0)
```

**Key:** The `filename` matches the case identifier in `Node_Matrix`; `source_node` and `target_node` refer to zero-based row positions *inside that case*, **not absolute worksheet row numbers**. The `edge_curve_id`, `source_face`, and `target_face` are STEP IDs, not PyG indices.

### 6.4 Geometry used for edge features

The code reconstructs approximate surface normals for planes, cylinders, spheres and cones where possible. For an edge shared by face `i` and face `j`, a representative edge point, normals `n_i, n_j`, and oriented tangent `t` are used. The approximate angle is:

\[
\theta_{ij}=\cos^{-1}(\mathrm{clip}(n_i\cdot n_j,-1,1))\,\frac{180}{\pi}.
\]

The signed convexity proxy is:

\[
s_{ij}=(n_i\times n_j)\cdot t.
\]

The generator uses its **own direction convention** to classify negative values as convex and positive as concave, with near-zero values assigned smooth. This sign depends on edge/face orientation conventions and should be checked on reference CAD examples before interpreting convexity physically.

For LINE, the length falls back to endpoint chord distance. For CIRCLE/ARC, the code computes an approximate radius-times-angle using endpoints; coincident endpoints are treated as a full circumference. **An arc's major/minor sweep may not always be determined correctly from two endpoints**, and unsupported curve types use a chord approximation. Thus `shared_edge_length` is not necessarily an exact CAD-kernel curve length.

If either surface normal or an oriented tangent cannot be determined, angle/sign values may be NaN and the text category may be `UNKNOWN`. Downstream GINE data preparation may fill selected missing numeric values with zero; distinguish **unknown** from a genuinely smooth edge when interpreting results.

### 6.5 Edge matrix versus graph edges

An undirected graph connection between face nodes `u` and `v` is represented in the binary adjacency by both `A[u,v]=1` and `A[v,u]=1`. The edge worksheet stores rows corresponding to **shared geometric B-Rep edges**, not necessarily one row per unique unordered node pair. A face pair can therefore generate multiple `Edge_Features` rows if it shares multiple edge IDs.

The within-family GINE loader duplicates each Excel edge row into two directed PyG entries `(u,v)` and `(v,u)` with the same edge attributes. It **does not consolidate duplicate face pairs**. The multi-seed GINE comparison, by contrast, averages multiple shared-edge feature rows for a face pair before aligning them to the adjacency-derived `edge_index`. These are meaningfully different aggregation choices; document the selected approach in experiment reports.

### 6.6 Effective GINE input channels

The within-family GINE code constructs a six-dimensional `edge_attr` from the raw Excel columns in this exact order:

| Feature index | Feature passed to GINE | Transformation |
|---:|---|---|
| 0 | `convexity_code` | Numeric code, NaNs filled with 0 |
| 1 | `dihedral_angle_norm` | `dihedral_angle_deg / 180` |
| 2 | `shared_length_norm` | `shared_edge_length / max(shared_edge_length within case)` |
| 3 | `edge_type_line` | Binary indicator |
| 4 | `edge_type_arc` | Binary indicator |
| 5 | `edge_type_spline` | Binary indicator |

In contrast, the `03-03_Model_Comparison_MultiSeed_GINE.py` script starts from the raw six columns (`convexity_code`, **degrees**, **length**, three edge-type indicators) and optionally standardizes them using training-set statistics. Its feature scale is **not identical** to the within-family GINE implementation by default.

## 7. Data preparation and PyTorch Geometric representation

### 7.1 Ordinary graph objects

Training code creates `torch_geometric.data.Data` objects:

```python
Data(
    x=...,          # Shape [num_faces, 11] or [num_faces, 8]
    edge_index=..., # Shape [2, num_directed_adjacencies]
    y=...,          # Shape [num_faces]
)
```

For edge-feature models:

```python
Data(
    x=...,          # Node features
    edge_index=..., # Directed face-neighbor entries
    edge_attr=...,  # One edge-feature vector per directed entry
    y=...,          # Node class labels
)
```

Each graph is constructed from a case block delimited by a blank worksheet row; batch training concatenates separate graphs with PyG's `DataLoader`. The `batch.ptr` array identifies boundaries between cases during strict case-level evaluation.

### 7.2 Coordinate-including versus coordinate-free layout

| Setting | `FEATURE_START_COL` | `FEATURE_END_COL` exclusive | `ADJ_START_COL` | Inputs | Label location |
|---|---:|---:|---:|---:|---|
| `USE_COORDINATE=True` | 1 | 13 | 14 | 11 | M (zero-based 12) |
| `USE_COORDINATE=False` | 1 | 10 | 11 | 8 | J (zero-based 9) |

To create an 8-feature coordinate-free workbook from an existing 11-feature workbook, one must remove the three `avg_x`, `avg_y`, and `avg_z` input columns **and preserve the label, blank separator, adjacency block, case boundaries, and—where applicable—`Edge_Features` sheet**. Neither uploaded `01_...` generator directly writes a `*_without_Coordinate.xlsx` or `*_noCoord_yesEdgeFeat.xlsx` file. These names are expected by later experiments and must be produced externally or by a separate preprocessing step.

For a coordinate-free layout the eight inputs are the four edge counts, three chord-length statistics, and `surface_type` integer code. If original positions `I:K` are deleted correctly, `J` becomes label, `K` blank, and `L` first adjacency column.

### 7.3 Truncated Excel adjacency columns

Some scripts reconstruct a square `num_nodes × num_nodes` adjacency when trailing all-empty Excel columns are omitted upon read. They take available adjacency columns, pad the remaining columns with zeros, symmetrize with `A = max(A, A.T)`, then set diagonal entries to zero. This works **only if enough of the missing values are recoverable from the retained transpose entries**. It cannot reconstruct arbitrary absent information. Verify the resulting matrix against the original STEP-derived graph.

### 7.4 Edge matching and data integrity

The GINE loaders match edge rows by file name; some normalize paths/case, while the multi-seed loader also strips extensions and whitespace. The multi-seed script exposes `STRICT_EDGE_MATCH=True`: with strict mode, it raises if an adjacency edge has no matching edge feature. `EDGE_NODE_ONE_BASED=None` attempts to infer whether source/target indices start at 0 or 1. **Avoid relying on inference if zero-based graph indices are known**; set `False` to prevent misinterpretation in graphs where node 0 simply has no incident edge rows.

## 8. Detailed script-by-script documentation

### 8.1 `01_FromSTEP_toExcel_FeatMatrixGenerated.py`

**Role:** a direct, sequential STEP-to-Excel generator without a subprocess intermediary. **Input:** STEP cases under `TestData/<Target>/`. **Output:** one ordinary workbook with node features and adjacency, no headers. **Default family:** `cantilever`.

Main procedures:

- `process_step(step_file)`: parse entity dictionary; identify faces; classify surface and edge types; calculate chord-length statistics and approximate coordinate averages; construct adjacency; apply fixed/load heuristic; return `(feature_df, adj_df)`.
- `main()`: iterate through STEP files, prefix each face with the original filename, insert a blank separator column and blank case row, pad varying adjacency widths, and write the final workbook.

Command:

```bash
python 01_FromSTEP_toExcel_FeatMatrixGenerated.py
```

Configuration is via `Target` inside the script. The file filter accepts `.step` and `.stp`. `Target.step` is excluded. There is no model training or confidence estimate in this generator.

### 8.2 `01_FromSTEP_toExcel_FeatMatrixGenerated_withEDGEFeature.py`

**Role:** create an edge-aware graph dataset suitable for GINE. **Input:** STEP files. **Output:** one two-sheet workbook. **Default family:** `Flange`.

- `process_step_file(step_file)`: retains the 11 face attributes/labels and extracts oriented shared-edge attributes including approximate normal angle and convexity.
- `build_dataset(target, base_dir)`: aggregates both sheets while maintaining case-level file identification.

Command:

```bash
python 01_FromSTEP_toExcel_FeatMatrixGenerated_withEDGEFeature.py --target Flange --base-dir .
```

**Important:** This generator currently filters `.step` only, unlike the node-only version. The direction/curvature computations are approximate and may contain unknown/missing classifications; review them before interpreting a GINE improvement as physically meaningful.

### 8.3 `02_Train_DesignFamily.py`

**Role:** within-family **GCN versus MLP** on one ordinary workbook. **Default input:** `TestDataResult_Excel\cantilever.xlsx`. It uses class labels `0,1,2`, `BATCH_SIZE=32`, `LEARNING_RATE=0.001`, `EPOCHS=1000`, `RANDOM_SEED=42`, and 60/20/20 case split. `VALIDATION_EXCEL_FILE=None` means automatic split. If set to a workbook, that file is actually used as the **final held-out test dataset**, while training/validation come from the main workbook.

Models: MLP 11→32→64→3 with ReLU, and a 2-layer GCN `11→32→64→3`. Evaluate node accuracy, node confusion matrix, and case-level strict accuracy. The current script trains both models each epoch and reports validation numbers, but final test selection uses the **last epoch** rather than a best-validation checkpoint.

Output: console prints plus timestamped `logs/YYYYMMDD_HHMMSS_Experiment.log`. The script does **not** explicitly export metrics to an Excel workbook or write a `.pt` checkpoint.

### 8.4 `02_Train_DesignFamily_withEDGEFeature.py`

**Role:** within-family **GINE versus MLP**, with adjacency and six shared-edge input channels. **Default input:** `TestDataResult_Excel\BookShelves_EDGE_Feat.xlsx`. Settings: `HIDDEN_DIM=64`, `DROPOUT=0.2` for GINE, `BATCH_SIZE=32`, `LEARNING_RATE=0.001`, `WEIGHT_DECAY=5e-4`, `EPOCHS=100`, seed 42, and 60/20/20 split.

- Parses `Node_Matrix` and `Edge_Features` by sheet names.
- Converts edge rows to directed PyG edges, with six derived attributes.
- Ensures no adjacency connection lacks an edge-feature pair, but may allow repeated directed edges if multiple B-Rep edges join a face pair.
- Trains a two-layer residual GINE with `BatchNorm1d`, ReLU, dropout, `train_eps=True` and a node-feature-only MLP baseline.
- Captures the best **validation node accuracy** state for each model and then evaluates validation and test outcomes.

Output: console/log file and a GINE state dictionary checkpoint at `best_gine_edge_model.pt` next to the Python script (includes dimensions, edge feature names and selected epoch). This is not an Excel results writer.

### 8.5 `03-01_Train_Feature_Adjacent_Matrix_without_XYZCoordinate.py`

**Role:** coordinate ablation using a prepared 8-feature workbook, comparing GCN with MLP. **Current input:** `TestDataResult_Excel\\LshapedBracket_without_Coordinate.xlsx`. `EPOCHS=300`, `BATCH_SIZE=32`, `LEARNING_RATE=0.001`, seed 42 and 60/20/20 case split. The code uses `FEATURE_START_COL=1`, `FEATURE_END_COL=10`, `ADJ_START_COL=11`. Its target is the last column of that feature slice.

Removing coordinates examines whether adjacency helps when absolute spatial positioning is unavailable. Do **not** just change column indices while keeping the original 11-feature workbook; first create a physically consistent 8-feature input workbook. Results are printed/logged; no new Excel output is created by this script.

### 8.6 `03-02_LOFO_Train4_TestCantilever.py`

**Role:** GCN leave-one-family-out test with a seen-family training/validation split and an untouched test family. **As configured:** `TRAIN_EXCEL_FILES=["Train_BoltShelvesFlangeLBrack.xlsx"]`, `TEST_EXCEL_FILES=["cantilever.xlsx"]`. Thus the name and current data settings agree: cantilever is held out, assuming the combined training workbook indeed contains exactly the four other families.

The experiment trains a two-layer GCN, uses internal seen-family validation (`VALID_RATIO=0.20`) and evaluates on the full held-out family. It reports node-level metrics, confusion matrix, per-class metrics, strict case accuracy, and incorrect case names. A checkpoint is intended at `gcn_lofo_test_cantilever.pt` next to the script.

**Important syntax issue in uploaded source:** line 21 has `from __future__ import annotations` *after* ordinary imports. Python requires future imports at the beginning of the module after the module docstring/comments, before other imports. **Move that line above `import logging` before running**. The provided file was inspected and fails `compile()` until corrected. This README does not silently change the uploaded code.

### 8.7 `03-02_LOFO_GINE_Edge_Train4_TestCantilever.py`

**Role:** LOFO experiment using GINE edge attributes and an MLP comparison. It retains an internal seen-family validation split and separate unseen-family evaluation. **Actual current configuration differs from the filename:**

```python
TRAIN_EXCEL_FILES = ["Train_Everything_Except_Flange.xlsx"]
TEST_EXCEL_FILES = ["Flange_EDGE_Feat.xlsx"]
MODEL_OUTPUT_FILE = SCRIPT_DIR / "gine_lofo_test_cantilever_edge.pt"
```

The configured held-out family is therefore **Flange**, not cantilever. The output checkpoint filename still mentions cantilever. To run a true cantilever LOFO experiment, change both the train and test Excel selections and ensure that the training workbook excludes cantilever. Prefer also renaming the checkpoint to match the actual held-out family. **Do not report current results as cantilever LOFO based solely on this script filename.**

Inputs are both two-sheet workbooks, each with `Node_Matrix` and `Edge_Features`. The GINE architecture and derived six edge channels are similar to the within-family edge experiment. The code stores a `.pt` checkpoint and logs/prints results.

### 8.8 `03-03_Model_Comparison_MultiSeed.py`

**Role:** fairer node-only model comparison across repeated random seeds. **Current workbook:** `BookShelves_without_Coordinate.xlsx`; `USE_COORDINATE=False`; `MODEL_LIST=["MLP","GCN","GraphSAGE","GIN"]`; `SEED_LIST=[0,1,2,3,4]`; `EPOCHS=200`; `BATCH_SIZE=32`; `LEARNING_RATE=0.001`; 60/20/20 split and `STANDARDIZE=True`.

The script creates a new split per seed but uses the *same seed-specific split* across the four models. Node features are standardized with training data only. Best model weights are selected using **validation macro-F1** rather than final-epoch accuracy. Test outputs include node accuracy, macro-F1, strict case accuracy, confusion matrix, and class recalls. Means and population standard deviations (`numpy.std()` default) are computed over five seeds.

CSV location is `RESULT_CSV = EXCEL_FILE.split(".")[0] + ".csv"`, so by default the summary sits beside the source workbook, with a matching base name. It contains one summary row per model, not each seed's detailed predictions; per-seed detail is in console/log output.

### 8.9 `03-03_Model_Comparison_MultiSeed_GINE.py`

**Role:** five-model repeated comparison including edge-aware GINE. **Current workbook:** `BookShelves_EDGE_Feat.xlsx`; `USE_COORDINATE=True`; `MODEL_LIST=["MLP","GCN","GraphSAGE","GIN","GINE"]`; seeds 0–4; `EPOCHS=200`; 60/20/20 split. Both node and edge standardization are enabled by default.

Inputs require the two-sheet edge workbook. Edge columns are normalized to canonical names through a column-alias helper. `STRICT_EDGE_MATCH=True` is the recommended integrity setting. Multiple geometric shared edges corresponding to one face pair are **averaged** to produce one feature vector for that graph connection. Both directions of the adjacency share that vector. Missing feature columns may be zero-filled with a warning, but strict mode raises if an adjacency connection itself has no edge row.

Same per-seed partition and model comparison framework is used as the node-only script. Test statistics include node accuracy, macro-F1 and strict accuracy. Output CSV:

```text
TestDataResult_Excel/BookShelves_EDGE_Feat_GINE_comparison.csv
```

The output is a **CSV**, not an Excel workbook. It contains columns including `excel_file`, `use_coordinate`, `edge_features`, `model`, `n_seeds`, means and standard deviations of the three metrics.

### 8.10 `04_Check_Graph_Topology.py`

**Role:** non-neural diagnostic for graph diversity and the empirical distinguishability of labels under node-only and Weisfeiler–Lehman-refined features. **Current settings:** `USE_COORDINATE=False`, five workbooks ending `*_noCoord_yesEdgeFeat.xlsx`, `WL_ITERATIONS=2`, `ROUND_DECIMALS=6`, `RESULT_CSV="topology_check_result.csv"`.

- `load_blocks()` parses the same first-sheet case-block layout and reconstructs symmetric adjacency.
- `structure_hash()` computes a degree-seeded WL-style graph fingerprint for counting distinct observed signatures.
- `wl_colors()` refines a node signature with its own prior signature and the sorted multiset of neighboring signatures for K rounds.
- `ceiling_from_groups()` groups observed nodes by signatures and totals the majority label in each group.
- `analyze_file()` reports counts, majority baseline, feature-only grouping score, K-round WL grouping score and difference.
- `main()` prints a summary and writes `topology_check_result.csv`.

**Careful interpretation:** The printed `distinct_topologies` counts **hash signatures**, not canonical graph-isomorphism classes, and hash collisions are possible. Similarly, the feature-only and WL values are **empirical label-consistency ceilings for the specific rounded input representations and available observed data**; calling them universal theoretical limits for all MLPs/GNNs would be too strong. They ignore edge attributes entirely, so the WL computation should **not** be presented as a bound on an edge-aware GINE model. Also, the code's original explanatory text equates one topology signature with no useful graph information; that inference is not universally valid—connectivity can help distinguish nodes even when all cases have the same abstract topology.

## 9. Experiments and evaluation metrics

### 9.1 Primary prediction task

Given a graph `G = (V,E)` and node features `X`, predict one of three boundary-condition labels `y_i ∈ {0,1,2}` for each face/node `i`. The implementations use supervised cross-entropy loss on node logits. This is **classification of faces**; it is not direct prediction of reaction force, pressure intensity, finite-element solution, or mesh displacement.

### 9.2 Node accuracy

\[
\mathrm{NodeAccuracy}=\frac{\sum_{i=1}^{N}\mathbf{1}(\widehat y_i=y_i)}{N}.
\]

All test faces contribute. For unequal-sized CAD cases, cases with more faces contribute more weight to this aggregate.

### 9.3 Strict case-level / geometry accuracy

\[
\mathrm{StrictCaseAccuracy}=\frac{1}{M}\sum_{g=1}^{M}
\mathbf{1}\bigl[\widehat{\mathbf y}_g=\mathbf y_g\bigr].
\]

A case is counted correct **only if every face label in the case is predicted correctly**. One incorrect face makes the entire case fail. Thus high node accuracy does not guarantee high strict case accuracy.

### 9.4 Confusion matrix and minority classes

Confusion matrix rows are true labels; columns are predicted labels, ordered `[0,1,2]` in the scripts. Diagonal entries are correct classifications. Class 0 (general) may dominate total node count; monitor fixed (1) and load (2) recall rather than only total accuracy.

\[
\mathrm{Recall}_{c}=\frac{TP_c}{TP_c+FN_c},\qquad
\mathrm{MacroF1}=\frac{1}{3}\sum_{c=0}^{2} F1_c.
\]

**Zero load recall** means no true load faces were correctly classified, irrespective of overall accuracy. Reporting `Node Accuracy + Macro-F1 + Recall per class + Strict Accuracy` is more informative than any single metric.

### 9.5 Model comparison

| Model | Node features | Binary adjacency | Edge attributes | Primary experimental role |
|---|---|---|---|---|
| MLP | Yes | No | No | Feature-only baseline |
| GCN | Yes | Yes | No | Normalized-neighbor message passing |
| GraphSAGE | Yes | Yes | No | Distinct self/neighbor transformations |
| GIN | Yes | Yes | No | Sum-aggregation topology-sensitive baseline |
| GINE | Yes | Yes | Yes | Relational geometry and edge-semantic ablation |

A GINE-versus-GIN difference can support the contribution of **edge-attribute representation under the implemented preprocessing/model choices**, but does not automatically establish general transfer to unseen CAD families.

### 9.6 Splits, checkpoints and selection criteria

For within-family scripts, a 60/20/20 case split is typical. In multi-seed scripts, each seed changes partitioning. In LOFO, all held-out-family cases are reserved for the final test; a validation subset is selected only among seen-family cases.

- `02_Train_DesignFamily.py` and `03-01...`: reports the final epoch model (no demonstrated best-checkpoint restoration in those versions).
- `02_Train_DesignFamily_withEDGEFeature.py`: restores best validation **node-accuracy** model.
- `03-03...` comparisons: selects model by validation **macro-F1**.
- LOFO scripts: inspect the selected metric and best-checkpoint implementation before comparing to different scripts; differences in evaluation/checkpoint rules should be disclosed.

## 10. Dataset combinations and ablation workflows

### 10.1 Basic node-only experiment

1. Generate `<Target>.xlsx` using the node-only 01 script.
2. Confirm its label column and adjacency are correct.
3. Point `02_Train_DesignFamily.py` to that file.
4. Run GCN/MLP training and record test node accuracy, confusion matrix and strict case accuracy.

### 10.2 Coordinate removal

1. Start from a complete node dataset.
2. Remove `avg_x`, `avg_y`, `avg_z` columns while retaining other features and labels.
3. Move labels/separator/adjacency consistently to the 8-feature Excel positions.
4. Save using a name such as `<Target>_without_Coordinate.xlsx`.
5. Run `03-01...` or the multi-seed node-only script with `USE_COORDINATE=False`.

Do not silently regenerate labels after removing coordinates. Preserve the original ground truth or manually verified labels.

### 10.3 Edge-semantic contribution

1. Generate `<Target>_EDGE_Feat.xlsx` from the edge-aware 01 script.
2. Verify the `Edge_Features` rows agree with adjacency for each case.
3. Run `02_Train_DesignFamily_withEDGEFeature.py` for GINE/MLP.
4. For a more controlled comparison run `03-03_Model_Comparison_MultiSeed_GINE.py` with all five models and aligned per-seed splits.
5. Report the comparison in terms of **MLP vs GIN** (binary relational structure) and **GIN vs GINE** (additional edge-feature information), qualifying architecture and preprocessing differences.

### 10.4 Leave-one-family-out experiment

```text
Seen family workbooks (four families) -> train + validation
Held-out family workbook (one family) -> final test only
```

The combined seen-family workbook must be assembled by an additional step; the provided scripts reference e.g. `Train_BoltShelvesFlangeLBrack.xlsx` or `Train_Everything_Except_Flange.xlsx` but do **not** generate these files themselves. For edge-aware LOFO, preserve **both** `Node_Matrix` and `Edge_Features` with consistent filename keys; mere stacking of node sheets is insufficient.

To change the held-out family, update `TRAIN_EXCEL_FILES`, `TEST_EXCEL_FILES`, and output filename together. Double-check source workbook membership so the same family never occurs on both sides.

### 10.5 Topology discriminability diagnostic

Set `EXCEL_FILES` and `USE_COORDINATE` to match the actual workbook layout; run `04_Check_Graph_Topology.py` and inspect `topology_check_result.csv`. Compare its observed feature-grouping and WL-grouping values with learned test results **as an exploratory diagnostic**. The diagnostic groups features using six-decimal rounding and uses the same source data to compute group-majority scores, so these quantities are not held-out test performance estimates.

## 11. Output artifacts and how to interpret them

### 11.1 Excel workbooks

| Artifact | Producer | Sheets | Headers | What it contains |
|---|---|---|---|---|
| `<Target>.xlsx` | Node 01 generator | One default sheet | No | Filename + 11 features + label + separator + adjacency, blank case rows |
| `<Target>_EDGE_Feat.xlsx` | Edge-aware 01 generator | `Node_Matrix` | No | Same node representation |
| Same edge workbook | Edge-aware 01 generator | `Edge_Features` | Yes | Filename + 14 edge-descriptor fields |
| `*_without_Coordinate.xlsx` | External/manual preprocessing | Usually one node sheet | Depends on preparation; expected no node headers | 8 node features + label + adjacency |
| `*_noCoord_yesEdgeFeat.xlsx` | External/manual preprocessing | Expected node and edge sheets for GINE | Node: no; Edge: yes | 8 node features plus preserved edge semantics |

**There is no produced Excel of prediction results in the uploaded training scripts.** The primary Excel outputs are datasets; training reports are console/log/CSV/checkpoint artifacts.

### 11.2 CSV results

**`03-03_Model_Comparison_MultiSeed.py` output columns:**

```text
excel_file, use_coordinate, model, n_seeds,
node_acc_mean, node_acc_std,
macro_f1_mean, macro_f1_std,
geometry_strict_mean, geometry_strict_std
```

**`03-03_Model_Comparison_MultiSeed_GINE.py`:** same summary fields plus `edge_features`. A row represents one model averaged across configured seeds. The `.csv` is suitable for statistical plotting, but does not contain per-case predictions or uncertainty intervals beyond the reported standard deviation.

**`04_Check_Graph_Topology.py` CSV:**

```text
file, use_coordinate, cases, nodes, distinct_topologies,
majority_baseline, mlp_ceiling_feature_only,
gnn_ceiling_wl2, structure_gain
```

The WL score header includes the configured iteration count, so it changes if `WL_ITERATIONS` changes. Values labeled "ceiling" are dataset/representation-dependent empirical grouping scores as explained above.

### 11.3 `.log` files

Many training scripts write timestamped logs in `logs/`, containing run time, script path, configured file names, hyperparameters and subsequent console/error output via a Tee stream. These logs are particularly important because some training scripts only print confusion matrices and case-level results without exporting them to tables. Archive the `.log` together with the corresponding source revision, input workbook, and random-seed settings.

### 11.4 `.pt` model checkpoints

The edge within-family trainer saves `best_gine_edge_model.pt`. The LOFO versions save `gcn_lofo_test_cantilever.pt` and `gine_lofo_test_cantilever_edge.pt` under their current configurations. `.pt` denotes a PyTorch checkpoint, **not** an Excel output. Inspect the checkpoint's metadata keys before reloading, use matching architecture and dimensions, and only load checkpoints from trusted sources.

### 11.5 Suggested dataset acceptance checks

Before publishing or training from an output workbook, verify:

- Number of STEP files processed equals number of nonempty case blocks.
- Number of face rows in each case equals the extracted `ADVANCED_FACE` count.
- Label values belong to `{0,1,2}` and reflect the intended FEA scenario.
- Feature columns contain finite numeric values; investigate categorical encoding and missing geometry.
- For a case of `n` faces, its adjacency is `n × n`, symmetric, binary, and has zero diagonal.
- Each `Edge_Features` `filename` matches a node case and each pair index lies in `[0, n-1]`.
- Each nonzero undirected adjacency pair has one or more corresponding geometric edge records.
- All expected sheets exist with correct header conventions.
- Coordinate-free datasets have the correct shifted label and adjacency column positions.
- Train and test family memberships are disjoint in LOFO.

## 12. Validation, reproducibility and limitations

### 12.1 Verification performed for this README

All ten supplied scripts were read from the uploaded files and subjected to Python **syntax compilation** (without executing training or STEP processing). Nine compiled successfully. The ordinary LOFO script failed because a future import is below other imports. This does not prove the nine compiling scripts are free of runtime or scientific issues, nor does it validate numerical results on the full CAD corpus.

### 12.2 Scientific limitations

1. **Labels are not universally identifiable from geometry alone.** Support/load conditions are functions of the engineering scenario, not just face shape. The generator's XYZ-based heuristic is a proposed label-assignment convention and must not be represented as verified physical ground truth for arbitrary parts.
2. **Surface and curve features are approximate.** Chord lengths, average vertex positions, normals, cone handling, full-circle handling and convexity signs require reference validation.
3. **One-line STEP entity assumptions.** The regex parser is not guaranteed to parse every syntactically valid STEP file.
4. **Topology preservation versus variation.** Parametric changes may leave topology constant, while hole-count and other topology-changing parameters can alter face/edge indexing and adjacency.
5. **Within-family and LOFO are different research questions.** Strong within-family accuracy does not imply robust unseen-family semantic transfer.
6. **The multi-seed results do not automatically establish statistical significance.** A paired statistical test and confidence intervals must be performed separately if claimed.
7. **Dataset correlations matter.** Parametric variants can be highly similar even when cases are partitioned independently. Report generation ranges and held-out parameter regimes if discussing generalization.
8. **Different trainers handle edge multiplicity and normalization differently.** Comparisons must explicitly state those differences.
9. **Hash/WL "ceilings" are diagnostics.** They depend on observed dataset labels, rounding, reference signatures and algorithmic assumptions; they are not exact generalization guarantees.
10. **No package-version manifest is included.** Reproducibility requires freezing dependencies (`pip freeze` or environment lockfile), Git commit identifiers, seeds, hardware/software versions and data hashes.

### 12.3 Pre-run issues found in source

| Script | Issue | Recommended action |
|---|---|---|
| `03-02_LOFO_Train4_TestCantilever.py` | `from __future__ import annotations` is below ordinary imports; **SyntaxError** | Move future import to first import position |
| `03-02_LOFO_GINE_Edge_Train4_TestCantilever.py` | Current held-out target is Flange although filename and checkpoint mention cantilever | Align config, file names and reported experimental label |
| `01_FromSTEP_toExcel_FeatMatrixGenerated.py` | Some comment/docstring remnants still contain Korean; STEP parsing is regex based | Translate remaining commentary separately; validate against representative STEP exporters |
| `03-01_Train_Feature_Adjacent_Matrix_without_XYZCoordinate.py` | Requires pre-generated 8-feature workbook | Do not feed an unmodified 11-feature workbook |
| `03-03_Model_Comparison_MultiSeed*.py` | Summary is CSV, not Excel; multiple standardization and checkpoint rules differ from `02` scripts | Preserve run configuration and describe comparison protocol |
| `04_Check_Graph_Topology.py` | Input files are configured as `*_noCoord_yesEdgeFeat.xlsx`, and code reads the first sheet without edge semantics | Confirm the first-sheet layout matches `USE_COORDINATE=False` |

## 13. Troubleshooting

**`FileNotFoundError: STEP folder not found`** — Check `Target`, relative project root, and folder spelling/casing. For the edge generator use `--base-dir` explicitly.

**`No STEP case files found`** — Confirm file extension and filter; node generator allows `.step`/`.stp`, edge generator currently allows `.step`. A file named exactly `<Target>.step` is excluded.

**`No graph case was loaded from Excel`** — Check that the worksheet has actual face rows and fully blank separators, that `header=None` is used for `Node_Matrix`, and that the correct sheet was selected.

**`adjacency shape ... expected (n,n)`** — Verify correct `ADJ_START_COL` and whether trailing all-empty Excel columns were omitted. Symmetry-based padding cannot restore arbitrary missing edge data.

**`Invalid edge node index`** — Match zero/one-based indexing and ensure `source_node`,`target_node` are local to the case. Verify face order was preserved between generator sheets.

**`adjacency edges without edge features`** — Confirm the `filename` key, sheet headers, source/target indices and shared-edge extraction. Do not globally disable strict matching simply to bypass unexpected missing geometry.

**`ModuleNotFoundError: torch_geometric`** — Install PyTorch Geometric appropriate for the environment after installing a compatible PyTorch build.

**`Expected ... features, found ...` or numeric NaN conversion errors** — Check that the input uses the appropriate 11-feature or 8-feature layout and that label/blank columns did not shift.

**`from __future__ imports must occur at the beginning of the file`** — Apply the import-order correction noted for the standard LOFO script.

**High node accuracy but zero case-level strict accuracy** — Inspect error distribution per case. All faces must be correct for a case to count. Also inspect minority Fixed/Load recall.

**Unexpectedly high accuracy on apparently unseen geometry** — Audit train/test case membership, duplicated STEP cases, feature/label leakage, and generator heuristics correlated with global coordinate orientation.

## 14. Quick-start commands

Run commands from the project root. Some scripts use hardcoded file paths or target names; **edit the configuration block before executing**.

```bash
# 1. Generate a node-only cantilever Excel dataset
python 01_FromSTEP_toExcel_FeatMatrixGenerated.py

# 2. Generate a Flange edge-aware Excel dataset
python 01_FromSTEP_toExcel_FeatMatrixGenerated_withEDGEFeature.py --target Flange --base-dir .

# 3. Within-family GCN and MLP (configure EXCEL_FILE first)
python 02_Train_DesignFamily.py

# 4. Within-family GINE and MLP (configure EXCEL_FILE first)
python 02_Train_DesignFamily_withEDGEFeature.py

# 5. Coordinate-free GCN and MLP (requires a derived 8-feature workbook)
python 03-01_Train_Feature_Adjacent_Matrix_without_XYZCoordinate.py

# 6. Standard LOFO — first fix the misplaced future import
python 03-02_LOFO_Train4_TestCantilever.py

# 7. Edge-aware LOFO — verify held-out family in TRAIN/TEST Excel lists
python 03-02_LOFO_GINE_Edge_Train4_TestCantilever.py

# 8. Node-only five-seed model comparison
python 03-03_Model_Comparison_MultiSeed.py

# 9. Edge-aware five-seed model comparison
python 03-03_Model_Comparison_MultiSeed_GINE.py

# 10. Graph topology/empirical grouping diagnostic
python 04_Check_Graph_Topology.py
```

## 15. Suggested methods description and citation notes

### 15.1 Condensed methods description (adapt to verified experiments)

> Parametric CAD parts were exported as STEP B-Rep models. Each `ADVANCED_FACE` was treated as a graph node and face adjacency was established by shared `EDGE_CURVE` references. Eleven face-level attributes were extracted, comprising edge-count descriptors, chord-length statistics, averaged extracted coordinate descriptors, and surface-type encoding. Binary adjacency matrices and face-wise boundary-condition labels were assembled into case-separated Excel workbooks. An edge-augmented representation additionally encoded shared-edge geometry through convexity, approximate dihedral angle, edge length and curve-type indicators. Within-family, coordinate-ablation, leave-one-family-out and multiple-seed training protocols were used to compare node-only and graph-based classifiers, with node-level and exact case-level accuracy as complementary evaluation criteria.

This wording intentionally does **not** claim a fully automatic or physically verified FEA boundary-condition assignment: label verification, case selection, external validation and numerical performance must be documented separately.

### 15.2 Scholarly attribution and licensing

The code imports PyTorch, PyTorch Geometric, scikit-learn, NumPy, pandas and openpyxl. Cite the actual library releases and relevant foundational methodology in the associated publication. For GIN/WL expressive-power discussion, the provided topology script refers to Xu et al., *How Powerful are Graph Neural Networks?* (ICLR 2019); verify full bibliographic details separately if placing the reference in a paper. This README does not establish a software license for the repository. Add `LICENSE` only with the rights-holder's approval and avoid inferring a license from a dependency.

---

**Maintenance recommendation:** version the generated data, configuration values, workbook schemas and code together. When revising the feature engineering or boundary-condition labeling convention, increment the dataset schema/version and regenerate dependent training/analysis outputs rather than silently reusing old workbooks.
