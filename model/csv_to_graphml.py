import os
import pandas as pd
import networkx as nx

# ---------------------------
# CONFIG: change if you want
# ---------------------------
TRAINING_MODE = True  # True when preparing train/val/test, False for real inference

# where your extracted PPVC folders live
EXTRACTED_ROOT = r"./Extracted Data"

# output graphml name inside each PPVC folder
GRAPHML_NAME = "graph.graphml"

# Map Revit category names -> element_type expected by GitHub ML
# (You can edit this freely)
CATEGORY_TO_ELEMENTTYPE = {
    "Walls": "Wall",
    "Structural Framing": "Beam",
    "Structural Foundation": "Foundation",
    "Floors": "Slab",
    "Generic Models": "GenericModel",
    "Rooms": "Zone",
    "Levels": "Level",
}

# Map annotation rows -> label_type (GitHub uses 5 classes)
# 0 = No Label
# 1 = Wall with Dimension
# 2 = Connected Label
# 3 = Door Marker
# 4 = Zone Stamp
#
# Adjust if your categories differ.
def annotation_row_to_label_type(row):
    cat = str(row.get("category", "")).lower()
    ann_type = str(row.get("annotation_type", "")).lower()
    val = str(row.get("value", "")).lower()

    # dimensions
    if "dimension" in cat:
        return 1

    # tags / markers
    if "tag" in cat:
        # if it looks like a door tag
        if "door" in ann_type or "door" in val:
            return 3
        return 2

    # text notes: usually not a label for elements, but you can map if needed
    if "text" in cat:
        return 2  # treat as connected label (edit if you want)

    # fallback
    return 2


# ---------------------------
# CORE LOGIC
# ---------------------------

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def normalize_element_type(category_name):
    if category_name in CATEGORY_TO_ELEMENTTYPE:
        return CATEGORY_TO_ELEMENTTYPE[category_name]
    # fallback: compress spaces to underscore
    return str(category_name).replace(" ", "_")


def build_label_map(annotation_df):
    """
    Returns dict: element_id -> label_type (int)
    We read target_element_ids column, which can be:
      "123|456|789"
    """
    label_map = {}

    if annotation_df is None or len(annotation_df) == 0:
        return label_map

    for _, row in annotation_df.iterrows():
        targets = str(row.get("target_element_ids", "")).strip()
        if not targets:
            continue

        lab = annotation_row_to_label_type(row)

        # split by | and assign
        for tid in targets.split("|"):
            tid = tid.strip()
            if not tid.isdigit():
                continue
            eid = int(tid)

            # if multiple labels hit same element,
            # keep the "strongest" (max)
            prev = label_map.get(eid, 0)
            label_map[eid] = max(prev, lab)

    return label_map


def csv_folder_to_graphml(folder_path):
    nodes_path = os.path.join(folder_path, "nodes.csv")
    edges_path = os.path.join(folder_path, "edges.csv")
    ann_path   = os.path.join(folder_path, "annotation.csv")

    if not (os.path.exists(nodes_path) and os.path.exists(edges_path)):
        print(f"[SKIP] Missing nodes/edges in {folder_path}")
        return

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    ann_df = None
    if TRAINING_MODE and os.path.exists(ann_path):
        ann_df = pd.read_csv(ann_path)

    label_map = build_label_map(ann_df) if TRAINING_MODE else {}


    G = nx.Graph()

    # ---- Add nodes ----
    for _, row in nodes_df.iterrows():
        eid = int(row["id"])

        category_name = str(row.get("category", "Unknown"))
        element_type = normalize_element_type(category_name)

        # features expected by repo (they will one-hot/encode)
        node_attrs = {
            "node_id": eid,                     # keep original id
            "element_type": element_type,       # string
            "category": category_name,          # string

            # bounding box
            "bb_xmin": safe_float(row.get("min_x")),
            "bb_ymin": safe_float(row.get("min_y")),
            "bb_zmin": safe_float(row.get("min_z")),
            "bb_xmax": safe_float(row.get("max_x")),
            "bb_ymax": safe_float(row.get("max_y")),
            "bb_zmax": safe_float(row.get("max_z")),

            # position (2D) for plotting / features
            "pos_x": safe_float(row.get("cx")),
            "pos_y": safe_float(row.get("cy")),

            # extra numeric features
            "length": safe_float(row.get("length")),
            "height": safe_float(row.get("height")),
            "thickness": safe_float(row.get("thickness")),
            "area": safe_float(row.get("area")),
            "width": safe_float(row.get("width")),
            "depth": safe_float(row.get("depth")),

            # string attrs (repo will ignore/zero them unless you encode later)
            "family": str(row.get("family", "")),
            "type_name": str(row.get("type", "")),
            "room_name": str(row.get("room_name", "")),
            "room_number": str(row.get("room_number", "")),

            # label
            "label_type": int(label_map.get(eid, 0))
        }

        # GraphML only likes basic types → force lists/dicts to str
        cleaned = {}
        for k, v in node_attrs.items():
            if isinstance(v, (list, dict)):
                cleaned[k] = str(v)
            else:
                cleaned[k] = v

        G.add_node(str(eid), **cleaned)

    # ---- Add edges (NO heuristics, only your edges.csv) ----
    for _, row in edges_df.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        etype = str(row.get("type", "adjacent"))

        if str(src) not in G.nodes or str(dst) not in G.nodes:
            continue

        G.add_edge(str(src), str(dst), type=etype)

    # ---- Save ----
    out_path = os.path.join(folder_path, GRAPHML_NAME)
    nx.write_graphml(G, out_path)
    print(f"[OK] Saved {out_path}  (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")


def main():
    # go through each PPVC subfolder
    for name in os.listdir(EXTRACTED_ROOT):
        folder = os.path.join(EXTRACTED_ROOT, name)
        if not os.path.isdir(folder):
            continue
        csv_folder_to_graphml(folder)


if __name__ == "__main__":
    main()
