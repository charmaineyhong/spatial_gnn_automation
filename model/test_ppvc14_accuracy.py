"""
Test PPVC 14 model accuracy by comparing expected vs actual predictions.
Uses 14_graph.graphml from 3.InputML and annotation_with_targets.csv from Extracted Data.
"""

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import networkx as nx
from GAT_4class_model import ImprovedGATModel
import sys

def annotation_row_to_label_type(row):
    """Convert annotation row to label_type (0=no_annotation, 1=dimension, 2=text, 3=both)"""
    category = str(row['category']).strip().lower()
    
    if 'dimension' in category:
        return 1  # dimension only
    elif 'text' in category:
        return 2  # text only
    else:
        return 0  # no_annotation

def create_expected_predictions(annotation_csv_path, nodes_csv_path, output_csv_path):
    """Create expected predictions CSV from annotation_with_targets.csv"""
    print(f"\n1. Creating expected predictions from {annotation_csv_path}")
    
    # Read nodes to get all element IDs
    nodes_df = pd.read_csv(nodes_csv_path)
    all_element_ids = nodes_df['id'].tolist()
    print(f"   Total nodes: {len(all_element_ids)}")
    
    # Read annotations
    annotations_df = pd.read_csv(annotation_csv_path)
    print(f"   Total annotations: {len(annotations_df)}")
    
    # Build mapping of element_id -> label_type
    element_to_label = {}
    
    for _, row in annotations_df.iterrows():
        annotation_id = row['annotation_id']
        category = row['category']
        label_type = annotation_row_to_label_type(row)
        
        # Get target elements
        target_ids_str = str(row.get('target_element_ids', ''))
        if target_ids_str and target_ids_str != 'nan':
            target_ids = [int(tid) for tid in target_ids_str.split('|') if tid.strip()]
            
            for target_id in target_ids:
                if target_id in element_to_label:
                    # If already has a label, upgrade to "both" if different
                    existing_label = element_to_label[target_id]
                    if existing_label != label_type:
                        element_to_label[target_id] = 3  # both
                else:
                    element_to_label[target_id] = label_type
    
    # Create predictions dataframe
    predictions = []
    for element_id in all_element_ids:
        predicted_label = element_to_label.get(element_id, 0)  # Default to no_annotation
        predictions.append({
            'element_id': element_id,
            'predicted_label': predicted_label
        })
    
    expected_df = pd.DataFrame(predictions)
    expected_df.to_csv(output_csv_path, index=False)
    
    # Print label distribution
    label_counts = expected_df['predicted_label'].value_counts().sort_index()
    print(f"\n   Expected label distribution:")
    label_names = {0: 'no_annotation', 1: 'dimension', 2: 'text', 3: 'both'}
    for label, count in label_counts.items():
        print(f"     {label} ({label_names[label]}): {count}")
    
    return expected_df

def run_model_inference(graphml_path, model_path, output_csv_path):
    """Run model inference on GraphML file"""
    print(f"\n2. Running model inference on {graphml_path}")
    
    # Load graph
    G = nx.read_graphml(graphml_path)
    print(f"   Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Extract node features (same as automated_inference_workflow.py)
    node_features = []
    node_ids = []
    node_mapping = {}
    
    for i, (node_id, data) in enumerate(G.nodes(data=True)):
        node_mapping[node_id] = i
        node_ids.append(int(node_id))  # Store as int for CSV
        
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
    
    # Build edge index
    edge_list = []
    for src, dst in G.edges():
        edge_list.append([node_mapping[src], node_mapping[dst]])
    
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"   Node features shape: {x.shape}")
    print(f"   Edge index shape: {edge_index.shape}")
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    
    # Load model
    device = torch.device('cpu')
    num_features = x.size(1)
    num_classes = 4
    
    model = ImprovedGATModel(
        input_dim=num_features,
        hidden_dim=128,
        num_heads=8,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   Model loaded: {num_features} features -> {num_classes} classes")
    
    # Run inference
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        probs = F.softmax(out, dim=1)
    
    # Create predictions dataframe
    predictions = []
    for i, element_id in enumerate(node_ids):
        predictions.append({
            'element_id': element_id,
            'predicted_label': pred[i].item()
        })
    
    actual_df = pd.DataFrame(predictions)
    actual_df.to_csv(output_csv_path, index=False)
    
    # Print label distribution
    label_counts = actual_df['predicted_label'].value_counts().sort_index()
    print(f"\n   Actual predictions distribution:")
    label_names = {0: 'no_annotation', 1: 'dimension', 2: 'text', 3: 'both'}
    for label, count in label_counts.items():
        print(f"     {label} ({label_names[label]}): {count}")
    
    return actual_df

def compare_predictions(expected_df, actual_df):
    """Compare expected vs actual predictions"""
    print(f"\n3. Comparing predictions")
    
    # Merge on element_id
    merged = expected_df.merge(actual_df, on='element_id', suffixes=('_expected', '_actual'))
    
    # Calculate accuracy
    total = len(merged)
    correct = (merged['predicted_label_expected'] == merged['predicted_label_actual']).sum()
    accuracy = correct / total * 100
    
    print(f"\n   Total elements: {total}")
    print(f"   Correct predictions: {correct}")
    print(f"   Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print(f"\n   Per-class accuracy:")
    label_names = {0: 'no_annotation', 1: 'dimension', 2: 'text', 3: 'both'}
    
    for label in sorted(label_names.keys()):
        class_mask = merged['predicted_label_expected'] == label
        class_total = class_mask.sum()
        
        if class_total > 0:
            class_correct = ((merged['predicted_label_expected'] == merged['predicted_label_actual']) & class_mask).sum()
            class_accuracy = class_correct / class_total * 100
            print(f"     {label} ({label_names[label]}): {class_correct}/{class_total} = {class_accuracy:.2f}%")
        else:
            print(f"     {label} ({label_names[label]}): 0/0 = N/A")
    
    # Show some mismatches
    mismatches = merged[merged['predicted_label_expected'] != merged['predicted_label_actual']]
    if len(mismatches) > 0:
        print(f"\n   First 10 mismatches:")
        print(mismatches[['element_id', 'predicted_label_expected', 'predicted_label_actual']].head(10).to_string(index=False))
    
    # Save comparison
    comparison_path = 'ppvc14_prediction_comparison.csv'
    merged.to_csv(comparison_path, index=False)
    print(f"\n   Full comparison saved to: {comparison_path}")
    
    return accuracy

def main():
    # Paths
    ppvc14_dir = r"c:\Users\charm\Spatial GNN\Extracted Data\PPVC 14_Typ"
    annotation_csv = f"{ppvc14_dir}\\annotation_with_targets.csv"
    nodes_csv = f"{ppvc14_dir}\\nodes.csv"
    graphml_path = r"c:\Users\charm\Spatial GNN\model\3.InputML\14_graph.graphml"
    model_path = r"c:\Users\charm\Spatial GNN\model\greattrained_model.pth"
    
    expected_csv = "ppvc14_expected_predictions.csv"
    actual_csv = "ppvc14_actual_predictions.csv"
    
    print("="*70)
    print("PPVC 14 Model Accuracy Test")
    print("="*70)
    
    # Step 1: Create expected predictions from annotations
    expected_df = create_expected_predictions(annotation_csv, nodes_csv, expected_csv)
    
    # Step 2: Run model inference
    actual_df = run_model_inference(graphml_path, model_path, actual_csv)
    
    # Step 3: Compare
    accuracy = compare_predictions(expected_df, actual_df)
    
    print("\n" + "="*70)
    print(f"FINAL ACCURACY: {accuracy:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
