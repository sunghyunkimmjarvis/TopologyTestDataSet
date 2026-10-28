"""STEP B-Rep -> face feature/adjacency matrix -> combined Excel.
Run: python 01_02_Integrated_STEP_to_Excel.py
Dependencies: numpy, pandas, openpyxl
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

# Change this single setting to select the CAD family.
Target = "cantilever"  # cantilever / Bolt / BookShelves / Flange / LshapedBracket

STEP_FOLDER = Path("TestData") / Target
FINAL_OUTPUT = Path("TestDataResult_Excel") / f"{Target}.xlsx"
FEATURE_COL_COUNT = 12  # 11 features + label


def process_step(step_file):
    """Return (feature_df, adjacency_df) for one STEP file."""
    STEP_FILE = step_file
    # ============================================================
    # STEP ASCII READ
    # ============================================================

    with open(STEP_FILE, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    print(f"STEP lines: {len(lines)}")

    # ============================================================
    # ENTITY DICTIONARY
    # ============================================================

    entity_dict = {}

    entity_pattern = re.compile(r'(#\d+)\s*=\s*(.*);')

    for line in lines:

        line = line.strip()

        match = entity_pattern.match(line)

        if match:

            entity_id = match.group(1)
            entity_body = match.group(2)

            entity_dict[entity_id] = entity_body

    print(f"Total entities: {len(entity_dict)}")

    # ============================================================
    # ADVANCED_FACE Extraction
    # ============================================================

    faces = []

    for entity_id, body in entity_dict.items():

        if body.startswith("ADVANCED_FACE"):

            faces.append({
                "face_id": entity_id,
                "body": body
            })

    print(f"Total ADVANCED_FACE: {len(faces)}")

    # ============================================================
    # Surface Type Extraction
    # ============================================================

    def extract_surface_type(face_body):

        refs = re.findall(r'#\d+', face_body)

        visited = set()
        stack = list(refs)

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)

            if current not in entity_dict:
                continue

            entity_text = entity_dict[current]

            if entity_text.startswith("PLANE"):
                return "PLANE"

            elif entity_text.startswith("CYLINDRICAL_SURFACE"):
                return "CYLINDRICAL"

            elif entity_text.startswith("SPHERICAL_SURFACE"):
                return "SPHERE"

            elif entity_text.startswith("CONICAL_SURFACE"):
                return "CONE"

            elif entity_text.startswith("TOROIDAL_SURFACE"):
                return "TORUS"

            elif entity_text.startswith("B_SPLINE_SURFACE"):
                return "NURBS"

            next_refs = re.findall(r'#\d+', entity_text)

            stack.extend(next_refs)

        return "ETC"

    # ============================================================
    # ADVANCED_FACE → EDGE_CURVE Extraction
    # ============================================================

    def get_edges_from_advanced_face(face_body):

        """
        ADVANCED_FACE
          -> FACE_BOUND
          -> EDGE_LOOP
          -> ORIENTED_EDGE
          -> EDGE_CURVE
        """

        edge_curve_ids = set()

        visited = set()

        stack = re.findall(r'#\d+', face_body)

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)

            if current not in entity_dict:
                continue

            entity_text = entity_dict[current]

            if entity_text.startswith("EDGE_CURVE"):

                edge_curve_ids.add(current)
                continue

            next_refs = re.findall(r'#\d+', entity_text)

            stack.extend(next_refs)

        return edge_curve_ids

    # ============================================================
    # EDGE TYPE Classification
    # ============================================================

    def classify_edge_curve(edge_curve_id):

        """
        EDGE_CURVE(#v1,#v2,#curve_geometry,.T.)

        curve_geometry 따라가서:
        LINE / CIRCLE / B_SPLINE_CURVE 판별
        """

        edge_text = entity_dict[edge_curve_id]

        refs = re.findall(r'#\d+', edge_text)

        for ref in refs:

            if ref not in entity_dict:
                continue

            geom_text = entity_dict[ref]

            if geom_text.startswith("LINE"):
                return "LINE"

            elif geom_text.startswith("CIRCLE"):
                return "ARC"

            elif geom_text.startswith("B_SPLINE_CURVE"):
                return "SPLINE"

        return "OTHER"

    # ============================================================
    # EDGE_CURVE endpoint vertex Coordinate Extraction
    # ============================================================
    def extract_edge_vertex_points(edge_curve_id):


        xyz_list = []

        edge_text = entity_dict[edge_curve_id]

        refs = re.findall(r'#\d+', edge_text)

        # 앞 두 개만 vertex
        vertex_refs = refs[:2]

        for vref in vertex_refs:

            if vref not in entity_dict:
                continue

            vertex_text = entity_dict[vref]

            if not vertex_text.startswith("VERTEX_POINT"):
                continue

            # VERTEX_POINT('',#123)
            vp_refs = re.findall(r'#\d+', vertex_text)

            if len(vp_refs) == 0:
                continue

            point_ref = vp_refs[0]

            if point_ref not in entity_dict:
                continue

            point_text = entity_dict[point_ref]

            if not point_text.startswith("CARTESIAN_POINT"):
                continue

            # Coordinate 부분만 Extraction
            coord_match = re.search(
                r'\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)',
                point_text
            )

            if coord_match:

                x = float(coord_match.group(1))
                y = float(coord_match.group(2))
                z = float(coord_match.group(3))

                xyz_list.append((x, y, z))

        return xyz_list

    # ============================================================
    # FULL CIRCLE CENTER Extraction
    # ============================================================

    def extract_full_circle_center(edge_curve_id):

        edge_text = entity_dict[edge_curve_id]

        refs = re.findall(r'#\d+', edge_text)

        circle_ref = None

        for ref in refs:

            if ref not in entity_dict:
                continue

            if entity_dict[ref].startswith("CIRCLE"):

                circle_ref = ref
                break

        if circle_ref is None:
            return None

        circle_text = entity_dict[circle_ref]

        circle_refs = re.findall(r'#\d+', circle_text)

        if len(circle_refs) == 0:
            return None

        axis_ref = circle_refs[0]

        if axis_ref not in entity_dict:
            return None

        axis_text = entity_dict[axis_ref]

        if not axis_text.startswith("AXIS2_PLACEMENT_3D"):
            return None

        axis_refs = re.findall(r'#\d+', axis_text)

        if len(axis_refs) == 0:
            return None

        center_ref = axis_refs[0]

        if center_ref not in entity_dict:
            return None

        point_text = entity_dict[center_ref]

        if not point_text.startswith("CARTESIAN_POINT"):
            return None

        coord_match = re.search(
            r'\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)',
            point_text
        )

        if coord_match:

            return (
                float(coord_match.group(1)),
                float(coord_match.group(2)),
                float(coord_match.group(3))
            )

        return None

    # ============================================================
    # EDGE LENGTH (Start-End Distance)
    # ============================================================

    def get_edge_chord_length(edge_curve_id):

        pts = extract_edge_vertex_points(edge_curve_id)

        if len(pts) < 2:
            return 0.0

        p1 = np.array(pts[0])
        p2 = np.array(pts[1])

        return float(
            np.linalg.norm(p2 - p1)
        )

    # ============================================================
    # Surface One-Hot Encoding
    # ============================================================

    surface_map = {
        "PLANE": 0,
        "CYLINDRICAL": 1,
        "SPHERE": 2,
        "CONE": 3,
        "TORUS": 4,
        "NURBS": 5,
        "ETC": 6
    }

    # ============================================================
    # Feature Matrix Generation
    # ============================================================

    feature_rows = []

    face_edges = {}

    for _, face in enumerate(faces):

        face_id = face["face_id"]

        face_body = face["body"]

        # --------------------------------------------------------
        # Surface Type
        # --------------------------------------------------------

        surface_type = extract_surface_type(face_body)

        surface_encoding = surface_map.get(
            surface_type,
            7
        )

        # --------------------------------------------------------
        # Edge Extraction
        # --------------------------------------------------------

        edge_ids = get_edges_from_advanced_face(face_body)

        face_edges[face_id] = edge_ids

        edge_types = [
            classify_edge_curve(edge_id)
            for edge_id in edge_ids
        ]

        total_edge_count = len(edge_types)

        line_edge_count = edge_types.count("LINE")

        arc_edge_count = edge_types.count("ARC")

        spline_edge_count = edge_types.count("SPLINE")

        # --------------------------------------------------------
        # Edge Length Statistics
        # --------------------------------------------------------

        edge_lengths = []

        for edge_id in edge_ids:

            dist = get_edge_chord_length(edge_id)

            if dist > 0:

                edge_lengths.append(dist)

        if len(edge_lengths) > 0:

            max_edge_length = round(max(edge_lengths), 4)

            min_edge_length = round(min(edge_lengths), 4)

            mean_edge_length = round(
                np.mean(edge_lengths),
                4
            )

        else:

            max_edge_length = 0.0
            min_edge_length = 0.0
            mean_edge_length = 0.0

        # --------------------------------------------------------
        # Vertex 평균 Coordinate
        # --------------------------------------------------------

        all_xyz = set()

        all_xyz = set()

        for edge_id in edge_ids:

            edge_xyz = extract_edge_vertex_points(edge_id)

            for p in edge_xyz:

                all_xyz.add(tuple(p))

            # ----------------------------------------------------
            # FULL CIRCLE이면 중심점 추가
            # ----------------------------------------------------

            if classify_edge_curve(edge_id) == "ARC":

                center = extract_full_circle_center(edge_id)

                if center is not None:

                    all_xyz.add(center)

        if len(all_xyz) > 0:

            xyz_array = np.array(list(all_xyz))

            avg_x = round(np.mean(xyz_array[:, 0]), 4)

            avg_y = round(np.mean(xyz_array[:, 1]), 4)

            avg_z = round(np.mean(xyz_array[:, 2]), 4)

        else:

            avg_x = 0.0
            avg_y = 0.0
            avg_z = 0.0

        # --------------------------------------------------------
        # Feature Row
        # --------------------------------------------------------

        row = [

            total_edge_count,
            line_edge_count,
            arc_edge_count,
            spline_edge_count,

            max_edge_length,
            min_edge_length,
            mean_edge_length,

            avg_x,
            avg_y,
            avg_z,

            surface_encoding,

            0
        ]

        feature_rows.append(row)

    # ============================================================
    # FEATURE MATRIX
    # ============================================================

    feature_columns = [

        "total_edge_count",
        "line_edge_count",
        "arc_edge_count",
        "spline_edge_count",

        "max_edge_length",
        "min_edge_length",
        "mean_edge_length",

        "avg_x",
        "avg_y",
        "avg_z",

        "surface_type",

        "label"
    ]

    feature_df = pd.DataFrame(
        feature_rows,
        columns=feature_columns
    )


    # ============================================================
    # ADJACENCY MATRIX
    # ============================================================

    n_faces = len(faces)

    adj_matrix = np.zeros((n_faces, n_faces), dtype=int)

    face_id_list = [f["face_id"] for f in faces]

    for i in range(n_faces):

        for j in range(i + 1, n_faces):

            shared_count = len(
                face_edges[face_id_list[i]]
                &
                face_edges[face_id_list[j]]
            )

            if shared_count > 0:

                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    adj_df = pd.DataFrame(adj_matrix)



    # ============================================================
    # Boundary Condition Label Automatic Generation
    # Note : You SHOULD Mend this labels by the intent of Engineer's FEM Analysis
    # ============================================================

    feature_df["avg_x"] = pd.to_numeric(feature_df["avg_x"])
    feature_df["avg_z"] = pd.to_numeric(feature_df["avg_z"])

    feature_df["label"] = 0

    fixed_face_idx = feature_df["avg_x"].idxmin()

    load_candidates = (
        feature_df["avg_z"]
        .sort_values(ascending=False)
        .index
    )

    load_face_idx = load_candidates[0]

    if load_face_idx == fixed_face_idx:
        load_face_idx = load_candidates[1]

    feature_df.loc[fixed_face_idx, "label"] = 1
    feature_df.loc[load_face_idx, "label"] = 2

    print(feature_df[["avg_x", "avg_z", "label"]])



    return feature_df, adj_df


def main():
    if not STEP_FOLDER.is_dir():
        raise FileNotFoundError(f"STEP folder not found: {STEP_FOLDER}")

    step_files = sorted(
        p for p in STEP_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in (".step", ".stp")
        and p.name.lower() != f"{Target}.step".lower()
    )
    if not step_files:
        raise FileNotFoundError(f"No STEP cases found in {STEP_FOLDER}")

    print(f"Total STEP files: {len(step_files)}")
    all_df = []
    for idx, step_file in enumerate(step_files, 1):
        print(f"[{idx}/{len(step_files)}] Processing: {step_file}")
        feature_df, adjacency_df = process_step(step_file)
        if feature_df.shape[1] != FEATURE_COL_COUNT:
            raise ValueError(f"Unexpected feature count in {step_file}")
        if adjacency_df.shape != (len(feature_df), len(feature_df)):
            raise ValueError(f"Invalid adjacency shape in {step_file}")

        # A: filename, B-L: features, M: label, N: blank, O+: adjacency
        merged = pd.concat(
            [feature_df.reset_index(drop=True),
             pd.DataFrame({"separator": [np.nan] * len(feature_df)}),
             adjacency_df.reset_index(drop=True)], axis=1
        )
        merged.insert(0, "filename", step_file.name)
        all_df.append(merged.to_numpy(dtype=object))
        all_df.append(np.full((1, merged.shape[1]), np.nan, dtype=object))

    # Different CAD cases may have different face counts (adjacency widths).
    width = max(block.shape[1] for block in all_df)
    padded = []
    for block in all_df:
        if block.shape[1] < width:
            block = np.pad(block, ((0, 0), (0, width - block.shape[1])),
                           mode="constant", constant_values=np.nan)
        padded.append(block)
    final_df = pd.DataFrame(np.vstack(padded))
    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_excel(FINAL_OUTPUT, index=False, header=False)
    print(f"DONE: {len(step_files)} cases saved to {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()
