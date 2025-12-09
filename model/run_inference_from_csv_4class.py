#!/usr/bin/env python3
"""
Single PPVC 4-class inference from nodes.csv + edges.csv

Usage (example):

  python run_inference_from_csv_4class.py ^
    --nodes_csv "C:/PPVC_AI/Test/nodes.csv" ^
    --edges_csv "C:/PPVC_AI/Test/edges.csv" ^
    --out_csv   "C:/PPVC_AI/Test/predictions.csv"
"""

import argparse
import os
# Force CPU mode to avoid CUDA library loading issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Fix numpy._core compatibility issue
import sys
import numpy as np
try:
    import numpy._core
except ImportError:
    # For older numpy versions, alias numpy.core to numpy._core
    sys.modules['numpy._core'] = np.core

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
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
# CSV Parsing for Revit-exported nodes/edges
# ----------------------------

def parse_csv_to_pyg(nodes_csv, edges_csv):
    """Parse nodes.csv and edges.csv into PyG Data object"""
    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)

    # Revit element Ids
    node_ids = df_nodes["id"].astype(int).tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    node_features = []

    for _, row in df_nodes.iterrows():
        # Basic geometry from NodeRecord CSV
        cx = float(row.get("cx", 0.0))
        cy = float(row.get("cy", 0.0))
        
        min_x = float(row.get("min_x", 0.0))
        max_x = float(row.get("max_x", 0.0))
        min_y = float(row.get("min_y", 0.0))
        max_y = float(row.get("max_y", 0.0))
        min_z = float(row.get("min_z", 0.0))
        max_z = float(row.get("max_z", 0.0))

        w = max_x - min_x
        h = max_y - min_y
        d = max_z - min_z

        # Fallbacks to avoid 0 dimensions
        if w <= 0: w = 1e-3
        if h <= 0: h = 1e-3
        if d <= 0: d = 1e-3

        # Match GraphML feature order exactly (20 features total)
        # Features 1-2: position
        x = cx
        y = cy
        
        # Features 3-5: bounding box min
        bb_xmin = min_x
        bb_ymin = min_y
        bb_zmin = min_z
        
        # Features 6-8: dimensions
        feats = [x, y, bb_xmin, bb_ymin, bb_zmin, w, h, d]

        # Features 9-10: volume & surface area
        volume = w * h * d
        surface_area = 2 * (w * h + h * d + w * d)
        feats.extend([volume, surface_area])

        # Features 11-13: ratios
        feats.extend([
            w / h if h else 0.0,
            h / d if d else 0.0,
            w / d if d else 0.0
        ])

        # Features 14-19: additional dimensions (use what we have, zeros for missing)
        length = w  # Use width as length
        height = h  # Use height
        thickness = d  # Use depth as thickness
        area = w * h  # Calculate area
        width = w
        depth = d
        feats.extend([length, height, thickness, area, width, depth])

        # Feature 20: element type encoding
        # Infer element type from category/type
        category = str(row.get("category", "")).lower()
        type_name = str(row.get("type", "")).lower()
        etype = "unknown"
        if "wall" in category:
            etype = "wall"
        elif "floor" in category or "slab" in category:
            etype = "slab"
        elif "structural framing" in category or "beam" in type_name:
            etype = "beam"
        elif "column" in category:
            etype = "column"
        
        type_encoding = {"wall": 1, "beam": 2, "foundation": 3, "slab": 4,
                        "genericmodel": 5, "zone": 6, "level": 7, "unknown": 0}
        feats.append(type_encoding.get(etype, 0))

        node_features.append(feats)

    # Edges (undirected) from src,dst columns
    edge_list = []
    for _, erow in df_edges.iterrows():
        try:
            src_id = int(erow["src"])
            dst_id = int(erow["dst"])
        except Exception:
            continue

        if src_id in id_to_idx and dst_id in id_to_idx:
            si = id_to_idx[src_id]
            di = id_to_idx[dst_id]
            edge_list.append([si, di])
            edge_list.append([di, si])

    if not edge_list:
        print(f"Warning: No edges in {nodes_csv}/{edges_csv}")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    x_tensor = torch.tensor(node_features, dtype=torch.float)

    print(
        f"  [CSV] Extracted {len(node_features)} nodes with "
        f"{len(node_features[0]) if node_features else 0} features"
    )

    data = Data(x=x_tensor, edge_index=edge_index)
    data.node_ids = node_ids
    
    return data, node_ids


# ----------------------------
# Model Loading
# ----------------------------

def load_model(model_path, input_dim, device):
    """Load the 4-class trained model"""
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    
    model = ImprovedGATModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_heads=8,
        num_classes=4,
        dropout=0.3
    ).to(device)
    
    try:
        # Try loading with weights_only=False first
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warning: Standard load failed, trying alternative method: {e}")
        # Alternative loading method
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


# ----------------------------
# Inference
# ----------------------------

def predict_4class(model, data, device):
    """
    Run inference with the 4-class model.
    Returns:
        predictions: numpy array of predicted classes (0, 1, 2, or 3)
        probabilities: numpy array of shape (num_nodes, 4) with class probabilities
    """
    data = data.to(device)
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
    
    return preds.cpu().numpy(), probs.cpu().numpy()


# ----------------------------
# Save Results
# ----------------------------

def save_predictions(node_ids, predictions, probabilities, output_path):
    """
    Save predictions to CSV file.
    
    Args:
        node_ids: List of element IDs
        predictions: Array of predicted classes (0-3)
        probabilities: Array of class probabilities (num_nodes, 4)
        output_path: Path to save CSV
    
    Returns:
        DataFrame with predictions
    """
    class_names = ["no_annotation", "dimension", "text", "both"]
    
    df = pd.DataFrame({
        'node_id': node_ids,
        'predicted_class': predictions,
        'confidence': probabilities.max(axis=1),
        'annotation_type': [class_names[p] for p in predictions]
    })
    
    df.to_csv(output_path, index=False)
    
    # Print summary statistics
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
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Single-file 4-class inference from CSV")
    parser.add_argument("--nodes_csv", required=True, help="Path to nodes.csv")
    parser.add_argument("--edges_csv", required=True, help="Path to edges.csv")
    parser.add_argument("--out_csv", required=True, help="Output predictions.csv path")

    parser.add_argument(
        "--model",
        default="./greattrained_model.pth",
        help="Path to 4-class model",
    )
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")

    args = parser.parse_args()

    # Force CPU to avoid CUDA library issues
    device = torch.device("cpu")

    try:
        print("=== CSV 4-Class GNN Inference ===")
        print("nodes_csv:", args.nodes_csv)
        print("edges_csv:", args.edges_csv)
        print("model:", args.model)
        print("device:", device)

        # 1) Build graph from CSV
        print("\n[1/4] Parsing CSV files...")
        data, node_ids = parse_csv_to_pyg(args.nodes_csv, args.edges_csv)
        input_dim = data.x.shape[1]
        print(f"✓ Loaded {len(node_ids)} nodes with {input_dim} features")

        # 2) Load model
        print("\n[2/4] Loading model...")
        model = load_model(args.model, input_dim, device)
        print("✓ Model loaded successfully")

        # 3) Run 4-class prediction
        print("\n[3/4] Running inference...")
        predictions, probabilities = predict_4class(model, data, device)
        print("✓ Inference complete")

        # 4) Save predictions
        print("\n[4/4] Saving results...")
        df = save_predictions(node_ids, predictions, probabilities, args.out_csv)
        print(f"✓ Predictions saved to: {args.out_csv}")

        print("\nFirst few predictions:")
        print(df.head())
        print("\n=== SUCCESS ===")
        
    except Exception as e:
        print(f"\n!!! ERROR !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
