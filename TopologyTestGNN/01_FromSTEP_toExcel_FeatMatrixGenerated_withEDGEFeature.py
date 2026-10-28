"""Create aggregated face-level and shared-edge STEP B-Rep datasets.

The Node_Matrix worksheet contains filename, eleven node features, a label,
a blank separator column, and an adjacency matrix per STEP case.
The Edge_Features worksheet contains shared-edge attributes and filenames.
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

Target = "Flange"  # Options: cantilever, Bolt, BookShelves, Flange, LshapedBracket.
FEATURE_COL_COUNT = 12  # Eleven face features and one boundary-condition label.
EDGE_COLUMNS = [
    "edge_curve_id", "source_face", "target_face",
    "source_node", "target_node", "convex_concave", "convexity_code",
    "dihedral_angle_deg", "shared_edge_length", "edge_type",
    "edge_type_line", "edge_type_arc", "edge_type_spline",
    "signed_convexity",
]


def process_step_file(step_file):
    """Extract node features, adjacency, and shared-edge features from one STEP file."""
    with open(step_file, "r", encoding="utf-8", errors="ignore") as f:
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
    # Extract ADVANCED_FACE entities
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
    # Extract surface types
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
    # Extract EDGE_CURVE references from ADVANCED_FACE
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
    # Extract ORIENTED_EDGE directions for each face
    # ============================================================
    def _step_bool(text, default=True):
        values = re.findall(r'\.(T|F)\.', text)
        if not values:
            return default
        return values[-1] == "T"


    def get_oriented_edges_from_face(face_body):
        """Return {EDGE_CURVE id: boundary direction (+1/-1)} for one face."""
        result = {}
        visited = set()
        stack = re.findall(r'#\d+', face_body)

        while stack:
            current = stack.pop()
            if current in visited or current not in entity_dict:
                continue
            visited.add(current)
            text = entity_dict[current]

            if text.startswith("ORIENTED_EDGE"):
                refs = re.findall(r'#\d+', text)
                edge_ref = next(
                    (r for r in reversed(refs)
                     if r in entity_dict and entity_dict[r].startswith("EDGE_CURVE")),
                    None,
                )
                if edge_ref is not None:
                    oriented_same = _step_bool(text, True)
                    edge_same = _step_bool(entity_dict[edge_ref], True)
                    result[edge_ref] = 1 if oriented_same == edge_same else -1
                continue

            stack.extend(re.findall(r'#\d+', text))

        return result


    # ============================================================
    # STEP geometry helpers for edge attributes
    # ============================================================
    def _extract_triplet(text):
        match = re.search(
            r'\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)',
            text,
        )
        if not match:
            return None
        return np.array([float(match.group(i)) for i in range(1, 4)], dtype=float)


    def get_cartesian_point(ref):
        text = entity_dict.get(ref, "")
        if not text.startswith("CARTESIAN_POINT"):
            return None
        return _extract_triplet(text)


    def get_direction(ref):
        text = entity_dict.get(ref, "")
        if not text.startswith("DIRECTION"):
            return None
        vec = _extract_triplet(text)
        if vec is None:
            return None
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-12 else None


    def get_axis2_placement(ref):
        text = entity_dict.get(ref, "")
        if not text.startswith("AXIS2_PLACEMENT_3D"):
            return None, None, None
        refs = re.findall(r'#\d+', text)
        origin = get_cartesian_point(refs[0]) if len(refs) > 0 else None
        axis = get_direction(refs[1]) if len(refs) > 1 else np.array([0.0, 0.0, 1.0])
        xdir = get_direction(refs[2]) if len(refs) > 2 else None
        return origin, axis, xdir


    def get_face_surface_ref(face_body):
        for ref in re.findall(r'#\d+', face_body):
            text = entity_dict.get(ref, "")
            if text.startswith((
                "PLANE", "CYLINDRICAL_SURFACE", "SPHERICAL_SURFACE",
                "CONICAL_SURFACE", "TOROIDAL_SURFACE", "B_SPLINE_SURFACE",
            )):
                return ref
        return None


    def get_face_normal(face_body, point):
        """Evaluate an approximate oriented surface normal at point."""
        surface_ref = get_face_surface_ref(face_body)
        if surface_ref is None or point is None:
            return None

        text = entity_dict[surface_ref]
        refs = re.findall(r'#\d+', text)
        normal = None

        if text.startswith("PLANE") and refs:
            _, axis, _ = get_axis2_placement(refs[0])
            normal = axis

        elif text.startswith("CYLINDRICAL_SURFACE") and refs:
            origin, axis, _ = get_axis2_placement(refs[0])
            if origin is not None and axis is not None:
                radial = point - origin - np.dot(point - origin, axis) * axis
                nr = np.linalg.norm(radial)
                normal = radial / nr if nr > 1e-12 else None

        elif text.startswith("SPHERICAL_SURFACE") and refs:
            origin, _, _ = get_axis2_placement(refs[0])
            if origin is not None:
                radial = point - origin
                nr = np.linalg.norm(radial)
                normal = radial / nr if nr > 1e-12 else None

        elif text.startswith("CONICAL_SURFACE") and refs:
            origin, axis, _ = get_axis2_placement(refs[0])
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
            semi_angle = float(nums[-1]) if nums else 0.0
            if origin is not None and axis is not None:
                rel = point - origin
                radial = rel - np.dot(rel, axis) * axis
                nr = np.linalg.norm(radial)
                if nr > 1e-12:
                    normal = radial / nr - np.tan(semi_angle) * axis

        if normal is None:
            return None

        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            return None
        normal = normal / norm

        # ADVANCED_FACE same_sense controls surface-normal orientation.
        if not _step_bool(face_body, True):
            normal = -normal
        return normal


    def get_circle_geometry(edge_curve_id):
        edge_text = entity_dict.get(edge_curve_id, "")
        for ref in re.findall(r'#\d+', edge_text):
            text = entity_dict.get(ref, "")
            if text.startswith("CIRCLE"):
                refs = re.findall(r'#\d+', text)
                if not refs:
                    return None
                origin, axis, xdir = get_axis2_placement(refs[0])
                nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
                radius = float(nums[-1]) if nums else None
                return origin, axis, xdir, radius
        return None


    def get_edge_midpoint(edge_curve_id):
        pts = extract_edge_vertex_points(edge_curve_id)
        if len(pts) >= 2:
            p1, p2 = np.array(pts[0]), np.array(pts[1])
            if np.linalg.norm(p2 - p1) > 1e-10:
                return 0.5 * (p1 + p2)
        circle = get_circle_geometry(edge_curve_id)
        if circle:
            origin, axis, xdir, radius = circle
            if origin is not None and radius is not None:
                if xdir is None:
                    trial = np.array([1.0, 0.0, 0.0])
                    if axis is not None and abs(np.dot(trial, axis)) > 0.9:
                        trial = np.array([0.0, 1.0, 0.0])
                    xdir = trial - np.dot(trial, axis) * axis
                    xdir /= np.linalg.norm(xdir)
                return origin + radius * xdir
        return None


    def get_edge_tangent(edge_curve_id, direction_sign=1):
        pts = extract_edge_vertex_points(edge_curve_id)
        if len(pts) >= 2:
            tangent = np.array(pts[1]) - np.array(pts[0])
            nt = np.linalg.norm(tangent)
            if nt > 1e-12:
                return direction_sign * tangent / nt

        circle = get_circle_geometry(edge_curve_id)
        if circle:
            _, axis, xdir, _ = circle
            if axis is not None and xdir is not None:
                tangent = np.cross(axis, xdir)
                nt = np.linalg.norm(tangent)
                if nt > 1e-12:
                    return direction_sign * tangent / nt
        return None


    def get_edge_true_length(edge_curve_id):
        edge_type = classify_edge_curve(edge_curve_id)
        pts = extract_edge_vertex_points(edge_curve_id)

        if edge_type == "ARC":
            circle = get_circle_geometry(edge_curve_id)
            if circle:
                origin, axis, _, radius = circle
                if origin is not None and axis is not None and radius is not None:
                    if len(pts) >= 2:
                        v1 = np.array(pts[0]) - origin
                        v2 = np.array(pts[1]) - origin
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 1e-12 and n2 > 1e-12 and np.linalg.norm(v2-v1) > 1e-10:
                            cosang = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
                            return float(radius * np.arccos(cosang))
                    return float(2.0 * np.pi * radius)

        # LINE and unsupported curves: endpoint chord; spline is an approximation.
        return get_edge_chord_length(edge_curve_id)


    def compute_shared_edge_attributes(face_i, face_j, edge_id, direction_sign):
        point = get_edge_midpoint(edge_id)
        n1 = get_face_normal(face_i["body"], point)
        n2 = get_face_normal(face_j["body"], point)
        tangent = get_edge_tangent(edge_id, direction_sign)

        angle = np.nan
        signed_value = np.nan
        convexity = "UNKNOWN"

        if n1 is not None and n2 is not None:
            angle = float(np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))))
            if tangent is not None:
                signed_value = float(np.dot(np.cross(n1, n2), tangent))
                if abs(signed_value) <= 1e-8 or angle <= 1e-6:
                    convexity = "SMOOTH"
                else:
                    # Convention based on oriented face boundary direction.
                    convexity = "CONVEX" if signed_value < 0.0 else "CONCAVE"

        edge_type = classify_edge_curve(edge_id)
        return {
            "edge_curve_id": edge_id,
            "source_face": face_i["face_id"],
            "target_face": face_j["face_id"],
            "source_node": face_id_list.index(face_i["face_id"]),
            "target_node": face_id_list.index(face_j["face_id"]),
            "convex_concave": convexity,
            "convexity_code": {"CONCAVE": -1, "SMOOTH": 0, "CONVEX": 1}.get(convexity, 0),
            "dihedral_angle_deg": round(angle, 6) if np.isfinite(angle) else np.nan,
            "shared_edge_length": round(get_edge_true_length(edge_id), 6),
            "edge_type": edge_type,
            "edge_type_line": 1 if edge_type == "LINE" else 0,
            "edge_type_arc": 1 if edge_type == "ARC" else 0,
            "edge_type_spline": 1 if edge_type == "SPLINE" else 0,
            "signed_convexity": round(signed_value, 8) if np.isfinite(signed_value) else np.nan,
        }

    # ============================================================
    # Classify edge geometry
    # ============================================================

    def classify_edge_curve(edge_curve_id):

        """
        EDGE_CURVE(#v1,#v2,#curve_geometry,.T.)

        Follow the curve geometry reference:
        Identify LINE, CIRCLE, or B_SPLINE_CURVE.
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
    # Extract EDGE_CURVE endpoint coordinates
    # ============================================================
    def extract_edge_vertex_points(edge_curve_id):

        """
        Extract the actual XYZ coordinates of
        the EDGE_CURVE endpoint vertices.
        """

        xyz_list = []

        edge_text = entity_dict[edge_curve_id]

        refs = re.findall(r'#\d+', edge_text)

        # Only the first two references are endpoint vertices.
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

            # Extract the coordinate tuple only.
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
    # Extract the center of a full circle
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
    # Build the feature matrix
    # ============================================================

    feature_rows = []

    face_edges = {}
    face_oriented_edges = {}

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
        # Extract edges
        # --------------------------------------------------------

        edge_ids = get_edges_from_advanced_face(face_body)

        face_edges[face_id] = edge_ids
        face_oriented_edges[face_id] = get_oriented_edges_from_face(face_body)

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
        # Average vertex coordinates
        # --------------------------------------------------------

        all_xyz = set()

        for edge_id in edge_ids:

            edge_xyz = extract_edge_vertex_points(edge_id)

            for p in edge_xyz:

                all_xyz.add(tuple(p))

            # ----------------------------------------------------
            # Include the center point for circular edges.
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
    # ADJACENCY MATRIX + SHARED EDGE FEATURE TABLE
    # ============================================================

    n_faces = len(faces)
    adj_matrix = np.zeros((n_faces, n_faces), dtype=int)
    face_id_list = [f["face_id"] for f in faces]
    face_lookup = {f["face_id"]: f for f in faces}
    edge_feature_rows = []

    for i in range(n_faces):
        for j in range(i + 1, n_faces):
            face_i_id = face_id_list[i]
            face_j_id = face_id_list[j]
            shared_edges = sorted(face_edges[face_i_id] & face_edges[face_j_id])

            if shared_edges:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

                for edge_id in shared_edges:
                    direction_sign = face_oriented_edges.get(face_i_id, {}).get(edge_id, 1)
                    edge_feature_rows.append(
                        compute_shared_edge_attributes(
                            face_lookup[face_i_id],
                            face_lookup[face_j_id],
                            edge_id,
                            direction_sign,
                        )
                    )

    adj_df = pd.DataFrame(adj_matrix)
    edge_feature_df = pd.DataFrame(edge_feature_rows, columns=[
        "edge_curve_id", "source_face", "target_face",
        "source_node", "target_node",
        "convex_concave", "convexity_code",
        "dihedral_angle_deg", "shared_edge_length",
        "edge_type", "edge_type_line", "edge_type_arc", "edge_type_spline",
        "signed_convexity",
    ])

    # ============================================================
    # Generate boundary-condition labels
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


    return feature_df, adj_df, edge_feature_df


def build_dataset(target, base_dir):
    """Aggregate every STEP case directly without copying files or subprocesses."""
    step_folder = base_dir / "TestData" / target
    output_file = base_dir / "TestDataResult_Excel" / f"{target}_EDGE_Feat.xlsx"
    if not step_folder.is_dir():
        raise FileNotFoundError(f"STEP directory does not exist: {step_folder}")

    step_files = sorted(
        path for path in step_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".step"
        and path.name.lower() != f"{target}.step".lower()
    )
    if not step_files:
        raise FileNotFoundError(f"No STEP case files found in: {step_folder}")

    print(f"Total STEP files: {len(step_files)}")
    node_frames = []
    edge_frames = []

    for index, step_file in enumerate(step_files, start=1):
        print(f"\n[{index}/{len(step_files)}] Processing: {step_file.name}")
        feature_df, adjacency_df, edge_df = process_step_file(step_file)

        if feature_df.empty:
            raise ValueError(f"No ADVANCED_FACE entities found in: {step_file}")

        # Preserve the original columns: A filename, B:L features,
        # M label, N blank separator, and O onward adjacency data.
        node_df = pd.concat(
            [
                feature_df.reset_index(drop=True),
                pd.DataFrame({"separator": [np.nan] * len(feature_df)}),
                adjacency_df.reset_index(drop=True),
            ],
            axis=1,
            ignore_index=True,
        )
        node_df.insert(0, "filename", step_file.name)
        node_frames.append(node_df)

        # Insert a blank row after each STEP case, as in the original script.
        node_frames.append(pd.DataFrame(np.nan, index=[0], columns=node_df.columns))

        edge_df = edge_df.copy()
        edge_df.insert(0, "filename", step_file.name)
        edge_frames.append(edge_df)
        print(f"Faces: {len(feature_df)}, shared edges: {len(edge_df)}")

    final_node_df = pd.concat(node_frames, ignore_index=True)
    final_edge_df = pd.concat(edge_frames, ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        final_node_df.to_excel(
            writer, sheet_name="Node_Matrix", index=False, header=False
        )
        final_edge_df.to_excel(
            writer, sheet_name="Edge_Features", index=False, header=True
        )

    print(f"\nSaved: {output_file}")
    print(f"Cases: {len(step_files)}")
    print(f"Node rows (including separators): {len(final_node_df)}")
    print(f"Shared-edge rows: {len(final_edge_df)}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=Target, help="CAD model family")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Project root")
    args = parser.parse_args()
    build_dataset(args.target, args.base_dir)
