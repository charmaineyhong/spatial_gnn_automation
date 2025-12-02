#!/usr/bin/env python3
"""
Combined Inference: Independent Dimension + Text Predictions

Both models run independently to predict annotation needs:
  - Dimension model: Predicts if element needs dimension annotations
  - Text model: Predicts if element needs text annotations

Final outputs:
  [0, 0] → No annotation needed
  [1, 0] → Dimension only
  [0, 1] → Text only
  [1, 1] → Both dimension AND text needed

Note: Text model was trained on elements WITHOUT dimensions, so predictions for
elements with dimensions may be less reliable. However, this approach allows
capturing elements that need both types of annotations.

Models used:
  - Dimension model: ./5.OutputML_GAT_DIM/trained_model.pth
  - Text model: ./5.OutputML_GAT_TEXT/trained_model_text.pth
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
# Model Definitions
# ----------------------------

class ImprovedGATModel(nn.Module):
    """Model for Dimension prediction (binary: 0=No Dim, 1=Has Dim)"""
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_classes=2, dropout=0.3):
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


class GATTextModel(nn.Module):
    """Model for Text prediction (binary: 0=No Text, 1=Has Text)"""
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_classes=2, dropout=0.3):
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
    Returns PyG Data object (without labels).
    """
    G = nx.read_graphml(graphml_path)
    
    node_features = []
    node_ids = []
    node_mapping = {}
    
    for i, (node_id, data) in enumerate(G.nodes(data=True)):
        node_mapping[node_id] = i
        node_ids.append(node_id)
        
        feats = []
        
        # Basic geometry
        x = float(data.get("pos_x", 0.0))
        y = float(data.get("pos_y", 0.0))
        z = float(data.get("bb_zmin", 0.0))
        w = float(data.get("width", 1.0))
        h = float(data.get("height", 1.0))
        d = float(data.get("depth", 1.0))
        feats.extend([x, y, z, w, h, d])
        
        # Volume & surface area
        volume = w * h * d
        surface_area = 2 * (w * h + h * d + w * d)
        feats.extend([volume, surface_area])
        
        # Ratios
        feats.extend([
            w / h if h else 0.0,
            h / d if d else 0.0,
            w / d if d else 0.0
        ])
        
        # Material & type encoding
        material = data.get("material", "unknown")
        element_type = data.get("type", "unknown")
        material_encoding = {"concrete": 1, "steel": 2, "wood": 3, "glass": 4, "unknown": 0}
        type_encoding = {"wall": 1, "column": 2, "beam": 3, "slab": 4, "unknown": 0}
        feats.extend([
            material_encoding.get(str(material).lower(), 0),
            type_encoding.get(str(element_type).lower(), 0),
        ])
        
        # Position again
        feats.extend([x, y, z])
        
        node_features.append(feats)
    
    # Edges (undirected)
    edge_list = []
    for src, dst in G.edges():
        if src in node_mapping and dst in node_mapping:
            edge_list.append([node_mapping[src], node_mapping[dst]])
            edge_list.append([node_mapping[dst], node_mapping[src]])
    
    if not edge_list:
        print(f"Warning: No edges in {graphml_path}")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    x_tensor = torch.tensor(node_features, dtype=torch.float)
    
    print(f"  Extracted {len(node_features)} nodes with {len(node_features[0]) if node_features else 0} features")
    
    return Data(x=x_tensor, edge_index=edge_index), node_ids


# ----------------------------
# CSV Parsing for Revit-exported nodes/edges
# ----------------------------

def parse_csv_to_pyg(nodes_csv, edges_csv):

    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)

    # Revit element Ids
    node_ids = df_nodes["id"].astype(int).tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    node_features = []

    material_encoding = {"concrete": 1, "steel": 2, "wood": 3, "glass": 4, "unknown": 0}
    type_encoding = {"wall": 1, "column": 2, "beam": 3, "slab": 4, "unknown": 0}

    for _, row in df_nodes.iterrows():
        # Basic geometry from NodeRecord CSV
        cx = float(row.get("cx", 0.0))
        cy = float(row.get("cy", 0.0))
        zmin = float(row.get("min_z", 0.0))

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

        x = cx
        y = cy
        z = zmin

        feats = [x, y, z, w, h, d]

        # Volume & surface area
        volume = w * h * d
        surface_area = 2 * (w * h + h * d + w * d)
        feats.extend([volume, surface_area])

        # Ratios
        feats.extend([
            w / h if h else 0.0,
            h / d if d else 0.0,
            w / d if d else 0.0
        ])

        # Infer material from family/type strings (very rough)
        family = str(row.get("family", "")).lower()
        type_name = str(row.get("type", "")).lower()
        material = "unknown"
        for key in ["concrete", "steel", "wood", "glass"]:
            if key in family or key in type_name:
                material = key
                break

        # Infer element type from category/type
        category = str(row.get("category", "")).lower()
        etype = "unknown"
        if "wall" in category:
            etype = "wall"
        elif "floor" in category or "slab" in category:
            etype = "slab"
        elif "structural framing" in category or "beam" in type_name:
            etype = "beam"
        elif "column" in category:
            etype = "column"

        feats.extend([
            material_encoding.get(material, 0),
            type_encoding.get(etype, 0),
        ])

        # Position again
        feats.extend([x, y, z])

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
    return data



# ----------------------------
# Inference
# ----------------------------

def load_models(dim_model_path, text_model_path, input_dim, device):
    """Load both trained models."""
    # Load dimension model checkpoint
    dim_checkpoint = torch.load(dim_model_path, map_location=device, weights_only=False)
    print(f"  Dimension model trained with input_dim={dim_checkpoint.get('input_dim')} (current: {input_dim})")
    dim_model = ImprovedGATModel(
        input_dim=input_dim,
        hidden_dim=dim_checkpoint.get('hidden_dim', 128),
        num_heads=dim_checkpoint.get('heads', 8),
        num_classes=2,
        dropout=dim_checkpoint.get('dropout', 0.3)
    ).to(device)
    dim_model.load_state_dict(dim_checkpoint['model_state_dict'])
    dim_model.eval()
    
    # Load text model checkpoint
    text_checkpoint = torch.load(text_model_path, map_location=device, weights_only=False)
    print(f"  Text model trained with input_dim={text_checkpoint.get('input_dim')} (current: {input_dim})")
    text_model = GATTextModel(
        input_dim=input_dim,
        hidden_dim=text_checkpoint.get('hidden_dim', 128),
        num_heads=text_checkpoint.get('heads', 8),
        num_classes=2,
        dropout=text_checkpoint.get('dropout', 0.35)
    ).to(device)
    text_model.load_state_dict(text_checkpoint['model_state_dict'])
    text_model.eval()
    
    print(f"  Dimension model: hidden_dim={dim_checkpoint.get('hidden_dim')}, heads={dim_checkpoint.get('heads')}")
    print(f"  Text model: hidden_dim={text_checkpoint.get('hidden_dim')}, heads={text_checkpoint.get('heads')}")
    
    if dim_checkpoint.get('input_dim') != input_dim or text_checkpoint.get('input_dim') != input_dim:
        print(f"  ⚠️  WARNING: Feature dimension mismatch! Models may not work correctly.")
    
    return dim_model, text_model


def predict_combined(dim_model, text_model, data, device, 
                     dim_threshold=0.5, text_threshold=0.5):
    """
    Run INDEPENDENT inference:
    1. Run dimension model - predicts if element needs dimensions
    2. Run text model - predicts if element needs text annotations
    3. Both predictions are independent and can both be true
    
    Args:
        dim_model: Dimension prediction model
        text_model: Text prediction model
        data: PyG Data object
        device: torch device
        dim_threshold: Threshold for dimension prediction
        text_threshold: Threshold for text prediction
    
    Returns:
        predictions: np.array of shape (num_nodes, 2) with [need_dim, need_text]
        probabilities: dict with 'dim' and 'text' probability arrays
    """
    data = data.to(device)
    
    with torch.no_grad():
        # Dimension predictions
        dim_out = dim_model(data.x, data.edge_index)
        dim_probs = F.softmax(dim_out, dim=1)
        dim_pred = (dim_probs[:, 1] >= dim_threshold).long()  # Class 1 = Has Dimension
        
        # Text predictions
        text_out = text_model(data.x, data.edge_index)
        text_probs = F.softmax(text_out, dim=1)
        text_pred = (text_probs[:, 1] >= text_threshold).long()  # Class 1 = Has Text
        
        # Debug info
        print(f"  Dimension: {dim_pred.sum().item()}/{len(dim_pred)} nodes predicted (prob range: {dim_probs[:, 1].min():.3f}-{dim_probs[:, 1].max():.3f})")
        print(f"  Text: {text_pred.sum().item()}/{len(text_pred)} nodes predicted (prob range: {text_probs[:, 1].min():.3f}-{text_probs[:, 1].max():.3f})")
        
        # Combine into [need_dim, need_text]
        predictions = torch.stack([dim_pred, text_pred], dim=1).cpu().numpy()
        
        probabilities = {
            'dim': dim_probs[:, 1].cpu().numpy(),
            'text': text_probs[:, 1].cpu().numpy()
        }
    
    return predictions, probabilities


def save_predictions(node_ids, predictions, probabilities, output_path):
    """Save predictions to CSV."""
    df = pd.DataFrame({
        'node_id': node_ids,
        'need_dimension': predictions[:, 0],
        'need_text': predictions[:, 1],
        'dim_probability': probabilities['dim'],
        'text_probability': probabilities['text'],
        'annotation_type': ['none' if p[0] == 0 and p[1] == 0 
                           else 'dimension_only' if p[0] == 1 and p[1] == 0
                           else 'text_only' if p[0] == 0 and p[1] == 1
                           else 'both' for p in predictions]
    })
    
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Print summary
    print("\n=== Prediction Summary ===")
    print(f"Total nodes: {len(node_ids)}")
    print(f"No annotation: {(df['annotation_type'] == 'none').sum()}")
    print(f"Dimension only: {(df['annotation_type'] == 'dimension_only').sum()}")
    print(f"Text only: {(df['annotation_type'] == 'text_only').sum()}")
    print(f"Both: {(df['annotation_type'] == 'both').sum()} (should be 0 in cascaded mode)")
    
    return df


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Combined Dimension + Text inference (cascaded)')
    parser.add_argument('--input', required=True, help='Path to input GraphML file')
    parser.add_argument('--output', default='combined_predictions.csv', 
                       help='Output CSV file path')
    parser.add_argument('--dim_model', default='./5.OutputML_GAT_DIM/trained_model.pth',
                       help='Path to dimension model')
    # Print summary
    print("\n=== Prediction Summary ===")
    print(f"Total nodes: {len(node_ids)}")
    print(f"No annotation: {(df['annotation_type'] == 'none').sum()}")
    print(f"Dimension only: {(df['annotation_type'] == 'dimension_only').sum()}")
    print(f"Text only: {(df['annotation_type'] == 'text_only').sum()}")
    print(f"Both: {(df['annotation_type'] == 'both').sum()}")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not os.path.exists(args.dim_model):
        raise FileNotFoundError(f"Dimension model not found: {args.dim_model}")
    print(f"=== Combined Inference (Independent Models) ===")
    print(f"Input: {args.input}")
    print(f"Dimension model: {args.dim_model}")
    print(f"Text model: {args.text_model}")
    print(f"Device: {args.device}")
    print(f"Thresholds: dim={args.dim_threshold}, text={args.text_threshold}")
    
    # Parse GraphML
    print("\nParsing GraphML...")
    data, node_ids = parse_graphml_to_pyg(args.input)
    input_dim = data.x.shape[1]
    print(f"Loaded {len(node_ids)} nodes with {input_dim} features")
    
    # Load models
    print("\nLoading models...")
    device = torch.device(args.device)
    dim_model, text_model = load_models(args.dim_model, args.text_model, input_dim, device)
    print("Models loaded successfully")
    
    # Run inference
    print("\nRunning inference...")
    predictions, probabilities = predict_combined(
        dim_model, text_model, data, device,
        args.dim_threshold, args.text_threshold
    )
    
    # Save results
    df = save_predictions(node_ids, predictions, probabilities, args.output)
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
