#!/usr/bin/env python3
"""
4-Class Inference: Single Model for All Annotation Types

This model predicts 4 classes directly:
  - Class 0: No annotation needed
  - Class 1: Dimension annotation only
  - Class 2: Text annotation only
  - Class 3: Both dimension AND text annotations needed

Model used:
  - 4-class model: ./4trained_model.pth
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
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
# GraphML Parsing
# ----------------------------

def parse_graphml_to_pyg(graphml_path):
    """
    Parse GraphML file and extract node features + edges.
    Returns PyG Data object (without labels) and list of node IDs.
    """
    G = nx.read_graphml(graphml_path)
    
    node_features = []
    node_ids = []
    node_mapping = {}
    
    for i, (node_id, data) in enumerate(G.nodes(data=True)):
        node_mapping[node_id] = i
        node_ids.append(node_id)
        
        features = []
        
        # Geometric features
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
    
    # Build edges
    edge_list = []
    for src, dst in G.edges():
        if src in node_mapping and dst in node_mapping:
            edge_list.append([node_mapping[src], node_mapping[dst]])
            edge_list.append([node_mapping[dst], node_mapping[src]])  # Undirected
    
    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    x_tensor = torch.tensor(node_features, dtype=torch.float)
    
    return Data(x=x_tensor, edge_index=edge_index), node_ids


# ----------------------------
# Model Loading
# ----------------------------

def load_model(model_path, input_dim, device):
    """Load the 4-class trained model"""
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
    parser = argparse.ArgumentParser(description='4-class annotation inference')
    parser.add_argument('--input', required=True,
                       help='Path to input GraphML file')
    parser.add_argument('--output', default=None,
                       help='Path to output CSV file (default: input_predictions.csv)')
    parser.add_argument('--model', default='./4trained_model.pth',
                       help='Path to 4-class trained model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = os.path.dirname(args.input) or '.'
        args.output = os.path.join(output_dir, f"{base_name}_predictions.csv")
    
    print(f"=== 4-Class Annotation Inference ===")
    print(f"Input GraphML: {args.input}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    
    # Check files exist
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Parse GraphML
    print(f"\nParsing GraphML...")
    data, node_ids = parse_graphml_to_pyg(args.input)
    print(f"✓ Loaded {len(node_ids)} nodes")
    print(f"✓ Node feature dimension: {data.x.shape[1]}")
    print(f"✓ Number of edges: {data.edge_index.shape[1]}")
    
    # Load model
    print(f"\nLoading model...")
    device = torch.device(args.device)
    model = load_model(args.model, data.x.shape[1], device)
    print(f"✓ Model loaded successfully")
    
    # Run inference
    print(f"\nRunning inference...")
    predictions, probabilities = predict_4class(model, data, device)
    print(f"✓ Inference complete")
    
    # Save results
    print(f"\nSaving predictions...")
    df = save_predictions(node_ids, predictions, probabilities, args.output)
    print(f"✓ Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
