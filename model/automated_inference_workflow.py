#!/usr/bin/env python3
"""
Automated workflow: CSV → Clean Annotations → GraphML → 4-Class Inference
Designed for Revit plugin integration

This script now includes the full cleaning pipeline:
1. Fix malformed annotation.csv (merge broken multi-line text)
2. Map text annotations to target elements using GraphML coordinates
3. Convert cleaned CSV to GraphML
4. Run 4-class GNN inference

This ensures inference uses the same data format as training.
"""

import argparse
import os
import sys
import math
import pandas as pd
import networkx as nx

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Fix numpy compatibility
try:
    import numpy._core
except ImportError:
    import numpy as np
    sys.modules['numpy._core'] = np.core

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# ----------------------------
# Model Definition
# ----------------------------

class ImprovedGATModel(nn.Module):
    """4-class GAT model for annotation prediction"""
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_classes=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim // num_heads, heads=num_heads,
                            dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                            dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                            dropout=dropout, concat=True)
        self.gat4 = GATConv(hidden_dim, hidden_dim // 2, heads=num_heads // 2,
                            dropout=dropout, concat=True)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm((hidden_dim // 2) * (num_heads // 2))

        final_dim = (hidden_dim // 2) * (num_heads // 2)
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        x = self.dropout(self.ln1(F.elu(self.gat1(x, edge_index))))
        x = self.dropout(self.ln2(F.elu(self.gat2(x, edge_index))))
        x = self.dropout(self.ln3(F.elu(self.gat3(x, edge_index))))
        x = self.ln4(F.elu(self.gat4(x, edge_index)))
        return self.classifier(x)


# ----------------------------
# Step 0: Clean Annotation CSV
# ----------------------------

def fix_annotation_csv(input_csv_path, output_csv_path):
    """
    Fix malformed annotation.csv by merging broken multi-line text values.
    
    The annotation.csv from Revit sometimes has unquoted newlines in the value field,
    causing rows to be split across multiple lines. This function detects and merges
    those continuation lines.
    
    Expected format: annotation_id,view_name,view_type,category,annotation_type,value,x,y,extra,target_element_ids
    """
    print("[Step 0a/4] Fixing malformed annotation CSV...")
    
    try:
        # Try reading with pandas first - if it works, CSV is already clean
        df = pd.read_csv(input_csv_path, on_bad_lines='skip', engine='python')
        
        # Clean up embedded newlines in value column
        if 'value' in df.columns:
            df['value'] = df['value'].astype(str).str.replace('\r', ' ').str.replace('\n', ' ').str.strip()
        
        df.to_csv(output_csv_path, index=False)
        print(f"  [OK] Fixed annotation CSV: {len(df)} rows")
        return output_csv_path
        
    except Exception as e:
        print(f"  Warning: pandas failed ({e}), falling back to manual parsing")
        
        # Manual line-by-line parsing for severely malformed CSV
        with open(input_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Detect expected column count from header
        header = lines[0].strip()
        expected_cols = len(header.split(','))
        
        fixed_lines = [header]
        buffer = ""
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # Count columns in current line
            cols = len(line.split(','))
            
            if cols < expected_cols:
                # This is a continuation line, merge with buffer
                buffer += " " + line
            else:
                # This is a complete line
                if buffer:
                    fixed_lines.append(buffer)
                buffer = line
        
        # Don't forget last buffered line
        if buffer:
            fixed_lines.append(buffer)
        
        # Write fixed CSV
        with open(output_csv_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        
        print(f"  [OK] Fixed annotation CSV: {len(fixed_lines) - 1} rows")
        return output_csv_path


def map_text_to_elements(annotation_csv, graphml_path, output_csv_path, max_distance=500.0, num_nearest=3):
    """
    Map text annotations to nearest elements using GraphML coordinates.
    
    This replicates the logic from map_text_to_elements.py used during training:
    - Load element coordinates from GraphML
    - For each TextNotes annotation, find nearest elements within max_distance
    - Fill in target_element_ids with pipe-separated list
    
    Args:
        annotation_csv: Path to cleaned annotation.csv
        graphml_path: Path to GraphML file with element coordinates
        output_csv_path: Output path for annotation_with_targets.csv
        max_distance: Maximum distance threshold for element mapping
        num_nearest: Number of nearest elements to assign per text annotation
    """
    print("[Step 0b/4] Mapping text annotations to target elements...")
    
    # Load GraphML and extract element coordinates
    if not os.path.exists(graphml_path):
        print(f"  Warning: GraphML not found at {graphml_path}, skipping text mapping")
        # Just copy the annotation file without mapping
        df = pd.read_csv(annotation_csv)
        df.to_csv(output_csv_path, index=False)
        return output_csv_path
    
    G = nx.read_graphml(graphml_path)
    elements = {}
    
    for node_id, data in G.nodes(data=True):
        x = float(data.get("pos_x", 0.0))
        y = float(data.get("pos_y", 0.0))
        elements[str(node_id)] = {"x": x, "y": y}
    
    print(f"  Loaded {len(elements)} elements from GraphML")
    
    # Load annotation CSV
    df = pd.read_csv(annotation_csv)
    
    # Convert numeric columns
    for col in ['x', 'y']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Track statistics
    text_rows = 0
    mapped_rows = 0
    
    # Process each row
    for idx, row in df.iterrows():
        category = str(row.get("category", "")).strip()
        value = str(row.get("value", "")).strip()
        
        # Check if this is a text annotation (handle both TextNote and TextNotes)
        if category.upper() in ["TEXTNOTES", "TEXTNOTE"] and value != "" and value != "nan":
            text_rows += 1
            
            # Get text coordinates
            try:
                text_x = float(row["x"])
                text_y = float(row["y"])
            except (ValueError, KeyError):
                continue
            
            # Find nearest elements
            distances = []
            for elem_id, elem_data in elements.items():
                ex = elem_data["x"]
                ey = elem_data["y"]
                dist = math.sqrt((ex - text_x)**2 + (ey - text_y)**2)
                distances.append((dist, elem_id))
            
            if distances:
                distances.sort()
                
                # Take top N nearest within threshold
                nearest = []
                for dist, elem_id in distances[:num_nearest]:
                    if dist <= max_distance:
                        nearest.append(elem_id)
                
                if nearest:
                    df.at[idx, "target_element_ids"] = "|".join(nearest)
                    mapped_rows += 1
    
    # Write output
    df.to_csv(output_csv_path, index=False)
    print(f"  [OK] Text annotations mapped: {mapped_rows}/{text_rows}")
    print(f"  [OK] Saved to: {output_csv_path}")
    
    return output_csv_path


# ----------------------------
# Step 1: Convert CSV to GraphML
# ----------------------------

def annotation_row_to_label_type(row):
    """
    Convert annotation row to label type:
    - 0: no annotation (shouldn't appear in annotation.csv but used as default)
    - 1: dimension annotation only
    - 2: text annotation only
    - 3: both dimension and text
    """
    category = str(row.get("category", "")).strip().upper()
    annotation_type = str(row.get("annotation_type", "")).strip().lower()
    
    # Dimension annotations
    if "DIMENSION" in category:
        return 1
    
    # Text annotations (handle both TextNote and TextNotes)
    if category in ["TEXTNOTES", "TEXTNOTE"]:
        return 2
    
    # Tags can be either (check annotation_type for clues)
    if category == "TAGS":
        if "dimension" in annotation_type:
            return 1
        elif "text" in annotation_type:
            return 2
    
    return 0


def build_label_map_from_annotations(annotation_csv):
    """
    Build mapping of element_id -> label_type from annotation_with_targets.csv
    
    Returns:
        dict: {element_id (str): label_type (int)}
    """
    if not os.path.exists(annotation_csv):
        return {}
    
    df = pd.read_csv(annotation_csv)
    label_map = {}
    
    for _, row in df.iterrows():
        targets = str(row.get("target_element_ids", "")).strip()
        if not targets or targets == 'nan':
            continue
        
        label_type = annotation_row_to_label_type(row)
        if label_type == 0:
            continue
        
        # Parse pipe-separated target element IDs
        for tid in targets.split("|"):
            tid = tid.strip()
            if not tid or not tid.replace('-', '').isdigit():
                continue
            
            # Merge labels: if element already has a different label, combine them
            prev = label_map.get(tid, 0)
            if prev == 0:
                label_map[tid] = label_type
            elif prev == label_type:
                label_map[tid] = label_type
            elif {prev, label_type} == {1, 2}:
                # Has both dimension and text
                label_map[tid] = 3
            else:
                # Keep the higher label (shouldn't happen often)
                label_map[tid] = max(prev, label_type)
    
    return label_map


def csv_to_graphml(nodes_csv, edges_csv, output_graphml, annotation_csv=None):
    """
    Convert CSV files to GraphML format matching training data structure.
    
    If annotation_csv is provided, reads target_element_ids to set label_type on nodes.
    """
    
    print("[Step 1/4] Converting CSV to GraphML...")
    
    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)
    
    # Build label map from annotations if provided
    label_map = {}
    if annotation_csv and os.path.exists(annotation_csv):
        print(f"  Loading annotation labels from: {os.path.basename(annotation_csv)}")
        label_map = build_label_map_from_annotations(annotation_csv)
        print(f"  Found {len(label_map)} elements with annotations")
    
    G = nx.DiGraph()
    
    category_map = {
        "Walls": "Wall",
        "Structural Framing": "Beam",
        "Structural Foundation": "Foundation",
        "Floors": "Slab",
        "Generic Models": "GenericModel",
        "Rooms": "Zone",
        "Levels": "Level",
    }
    
    for _, row in df_nodes.iterrows():
        node_id = str(row['id'])
        category = str(row.get('category', 'Unknown'))
        element_type = category_map.get(category, "Unknown")
        
        node_attrs = {
            'node_id': node_id,
            'element_type': element_type,
            'category': category,
            'bb_xmin': float(row.get('min_x', 0.0)),
            'bb_ymin': float(row.get('min_y', 0.0)),
            'bb_zmin': float(row.get('min_z', 0.0)),
            'bb_xmax': float(row.get('max_x', 0.0)),
            'bb_ymax': float(row.get('max_y', 0.0)),
            'bb_zmax': float(row.get('max_z', 0.0)),
            'pos_x': float(row.get('cx', 0.0)),
            'pos_y': float(row.get('cy', 0.0)),
            'length': float(row.get('length', 0.0)),
            'height': float(row.get('height', 0.0)),
            'thickness': float(row.get('thickness', 0.0)),
            'area': float(row.get('area', 0.0)),
            'width': float(row.get('width', 0.0)),
            'depth': float(row.get('depth', 0.0)),
            'family': str(row.get('family', '')),
            'type_name': str(row.get('type', '')),
            'room_name': str(row.get('room_name', '')),
            'room_number': str(row.get('room_number', '')),
        }
        
        # Add label_type from annotation mapping
        if node_id in label_map:
            node_attrs['label_type'] = label_map[node_id]
        else:
            node_attrs['label_type'] = 0  # No annotation
        
        G.add_node(node_id, **node_attrs)
    
    for _, row in df_edges.iterrows():
        src = str(row['src'])
        dst = str(row['dst'])
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst)
    
    nx.write_graphml(G, output_graphml, encoding='utf-8')
    
    # Print label statistics
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, data in G.nodes(data=True):
        label = data.get('label_type', 0)
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"  [OK] GraphML created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Label distribution: no_annotation={label_counts[0]}, dimension={label_counts[1]}, text={label_counts[2]}, both={label_counts[3]}")
    
    return output_graphml


# ----------------------------
# Step 2: Parse GraphML to PyG Data
# ----------------------------

def parse_graphml_to_pyg(graphml_path):
    """Parse GraphML file and extract node features + edges"""
    
    print("[Step 2/4] Parsing GraphML...")
    
    G = nx.read_graphml(graphml_path)
    
    node_features = []
    node_ids = []
    node_mapping = {}
    
    for i, (node_id, data) in enumerate(G.nodes(data=True)):
        node_mapping[node_id] = i
        node_ids.append(node_id)
        
        features = []
        
        x = float(data.get("pos_x", 0.0))
        y = float(data.get("pos_y", 0.0))

        bb_xmin = float(data.get("bb_xmin", 0.0))
        bb_ymin = float(data.get("bb_ymin", 0.0))
        bb_zmin = float(data.get("bb_zmin", 0.0))
        bb_xmax = float(data.get("bb_xmax", 0.0))
        bb_ymax = float(data.get("bb_ymax", 0.0))
        bb_zmax = float(data.get("bb_zmax", 0.0))

        w = bb_xmax - bb_xmin if bb_xmax > bb_xmin else 1.0
        h = bb_ymax - bb_ymin if bb_ymax > bb_ymin else 1.0
        d = bb_zmax - bb_zmin if bb_zmax > bb_zmin else 1.0

        features.extend([x, y, bb_xmin, bb_ymin, bb_zmin, w, h, d])

        volume = w * h * d
        surface_area = 2 * (w * h + h * d + w * d)
        features.extend([volume, surface_area])

        features.extend([
            w / h if h else 0.0,
            h / d if d else 0.0,
            w / d if d else 0.0,
        ])

        length = float(data.get("length", 0.0))
        height = float(data.get("height", 0.0))
        thickness = float(data.get("thickness", 0.0))
        area = float(data.get("area", 0.0))
        width = float(data.get("width", 0.0))
        depth = float(data.get("depth", 0.0))
        features.extend([length, height, thickness, area, width, depth])

        element_type = data.get("element_type", "unknown")
        type_encoding = {"wall": 1, "beam": 2, "foundation": 3, "slab": 4,
                        "genericmodel": 5, "zone": 6, "level": 7, "unknown": 0}
        features.append(type_encoding.get(str(element_type).lower(), 0))
        
        node_features.append(features)
    
    edge_list = []
    for src, dst in G.edges():
        if src in node_mapping and dst in node_mapping:
            edge_list.append([node_mapping[src], node_mapping[dst]])
            edge_list.append([node_mapping[dst], node_mapping[src]])
    
    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    x_tensor = torch.tensor(node_features, dtype=torch.float)
    
    print(f"  [OK] Loaded {len(node_ids)} nodes with {len(node_features[0])} features")
    
    return Data(x=x_tensor, edge_index=edge_index), node_ids


# ----------------------------
# Step 3: Run Inference
# ----------------------------

def load_model(model_path, input_dim, device):
    """Load the 4-class trained model"""
    import warnings
    warnings.filterwarnings('ignore')
    
    model = ImprovedGATModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_heads=8,
        num_classes=4,
        dropout=0.3
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def predict_4class(model, data, device):
    """Run inference with the 4-class model"""
    data = data.to(device)
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
    
    return preds.cpu().numpy(), probs.cpu().numpy()


def save_predictions(node_ids, predictions, probabilities, output_path):
    """Save predictions to CSV file"""
    class_names = ["no_annotation", "dimension", "text", "both"]
    
    df = pd.DataFrame({
        'node_id': node_ids,
        'predicted_class': predictions,
        'confidence': probabilities.max(axis=1),
        'annotation_type': [class_names[p] for p in predictions]
    })
    
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Prediction Summary:")
    print(f"{'='*60}")
    print(f"Total elements: {len(node_ids)}")
    for i, name in enumerate(class_names):
        count = (predictions == i).sum()
        pct = 100 * count / len(predictions)
        print(f"  {name:20s}: {count:5d} ({pct:5.1f}%)")
    print(f"{'='*60}")
    
    return df


# ----------------------------
# Main Workflow
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Automated CSV to 4-class inference workflow with annotation cleaning')
    parser.add_argument('--nodes_csv', required=True, help='Path to nodes.csv')
    parser.add_argument('--edges_csv', required=True, help='Path to edges.csv')
    parser.add_argument('--annotation_csv', help='Path to annotation.csv (optional, for cleaning)')
    parser.add_argument('--out_csv', required=True, help='Output predictions CSV path')
    parser.add_argument('--model', default='./mostlatesttrained_model.pth', help='Path to trained model')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--keep_graphml', action='store_true', help='Keep intermediate GraphML file')
    parser.add_argument('--save_cleaned', action='store_true', help='Save cleaned annotation and GraphML to source folder')
    parser.add_argument('--max_distance', type=float, default=500.0, help='Max distance for text-to-element mapping')
    parser.add_argument('--num_nearest', type=int, default=3, help='Number of nearest elements per text annotation')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    try:
        print("="*60)
        print("=== Automated 4-Class GNN Inference Workflow ===")
        print("="*60)
        print(f"Nodes CSV: {args.nodes_csv}")
        print(f"Edges CSV: {args.edges_csv}")
        if args.annotation_csv:
            print(f"Annotation CSV: {args.annotation_csv}")
        print(f"Model: {args.model}")
        print(f"Output: {args.out_csv}")
        print()
        
        # Create temporary directory for intermediate files
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_graphml = os.path.join(temp_dir, "temp_inference.graphml")
        
        # Step 0: Clean annotation CSV (if provided)
        if args.annotation_csv:
            # Get source folder for saving cleaned files
            source_folder = os.path.dirname(os.path.abspath(args.annotation_csv))
            
            temp_fixed_annotation = os.path.join(temp_dir, "temp_annotation_fixed.csv")
            temp_annotation_with_targets = os.path.join(temp_dir, "temp_annotation_with_targets.csv")
            
            # Step 0a: Fix malformed CSV
            fix_annotation_csv(args.annotation_csv, temp_fixed_annotation)
            
            # Step 0b: Create initial GraphML (without labels) to get element coordinates
            temp_graphml_unlabeled = os.path.join(temp_dir, "temp_inference_unlabeled.graphml")
            csv_to_graphml(args.nodes_csv, args.edges_csv, temp_graphml_unlabeled, annotation_csv=None)
            
            # Step 0c: Map text annotations to elements using GraphML coordinates
            map_text_to_elements(temp_fixed_annotation, temp_graphml_unlabeled, temp_annotation_with_targets, 
                               args.max_distance, args.num_nearest)
            
            # Step 1: Recreate GraphML with labels from cleaned annotations
            csv_to_graphml(args.nodes_csv, args.edges_csv, temp_graphml, annotation_csv=temp_annotation_with_targets)
            
            print(f"  [OK] Annotation cleaning and labeling complete")
            
            # Save cleaned files to source folder if requested
            if args.save_cleaned:
                cleaned_annotation_path = os.path.join(source_folder, "annotation_with_targets.csv")
                cleaned_graphml_path = os.path.join(source_folder, "graph.graphml")
                
                import shutil
                shutil.copy2(temp_annotation_with_targets, cleaned_annotation_path)
                shutil.copy2(temp_graphml, cleaned_graphml_path)
                
                print(f"  [OK] Saved cleaned annotation to: {cleaned_annotation_path}")
                print(f"  [OK] Saved labeled GraphML to: {cleaned_graphml_path}")
            
            # Cleanup intermediate unlabeled GraphML
            try:
                os.remove(temp_graphml_unlabeled)
            except:
                pass
            
        else:
            # No annotation cleaning needed, just convert CSV to GraphML
            csv_to_graphml(args.nodes_csv, args.edges_csv, temp_graphml, annotation_csv=None)
        
        # Step 2: Parse GraphML to PyG Data
        data, node_ids = parse_graphml_to_pyg(temp_graphml)
        input_dim = data.x.shape[1]
        
        # Step 3: Load model and run inference
        print("[Step 3/4] Running inference...")
        model = load_model(args.model, input_dim, device)
        print(f"  [OK] Model loaded")
        
        predictions, probabilities = predict_4class(model, data, device)
        print(f"  [OK] Inference complete")
        
        # Step 4: Save results
        print("[Step 4/4] Saving predictions...")
        df = save_predictions(node_ids, predictions, probabilities, args.out_csv)
        print(f"\n[OK] Predictions saved to: {args.out_csv}")
        
        # Cleanup temporary files
        if not args.keep_graphml:
            try:
                os.remove(temp_graphml)
                if args.annotation_csv:
                    os.remove(temp_fixed_annotation)
                    os.remove(temp_annotation_with_targets)
            except:
                pass
        
        print("\n=== SUCCESS ===")
        
    except Exception as e:
        print(f"\n!!! ERROR !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
