#!/usr/bin/env python3
"""
Batch 4-Class Inference: Process all GraphML files in a directory

Runs 4-class annotation inference on all GraphML files.
"""

import os
import argparse
import glob
from inference_4class import (
    parse_graphml_to_pyg,
    load_model,
    predict_4class,
    save_predictions
)
import torch


def main():
    parser = argparse.ArgumentParser(description='Batch 4-class inference on all GraphML files')
    parser.add_argument('--input_dir', default='./3.InputML',
                       help='Directory containing GraphML files')
    parser.add_argument('--output_dir', default='./6.Predictions_4Class',
                       help='Output directory for predictions')
    parser.add_argument('--model', default='./4trained_model.pth',
                       help='Path to 4-class trained model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all GraphML files
    graphml_files = sorted(glob.glob(os.path.join(args.input_dir, '*.graphml')))
    
    if not graphml_files:
        print(f"No GraphML files found in {args.input_dir}")
        return
    
    print(f"=== Batch 4-Class Annotation Inference ===")
    print(f"Found {len(graphml_files)} GraphML files")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    
    # Load model once
    print("\nLoading model...")
    # Parse first file to get input_dim
    data_temp, _ = parse_graphml_to_pyg(graphml_files[0])
    input_dim = data_temp.x.shape[1]
    
    device = torch.device(args.device)
    model = load_model(args.model, input_dim, device)
    print("✓ Model loaded successfully\n")
    
    # Process each file
    results_summary = []
    
    for i, graphml_path in enumerate(graphml_files, 1):
        filename = os.path.basename(graphml_path)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_predictions.csv")
        
        print(f"[{i}/{len(graphml_files)}] Processing {filename}...")
        
        try:
            # Parse GraphML
            data, node_ids = parse_graphml_to_pyg(graphml_path)
            print(f"  Loaded {len(node_ids)} nodes")
            
            # Run inference
            predictions, probabilities = predict_4class(model, data, device)
            
            # Save predictions
            df = save_predictions(node_ids, predictions, probabilities, output_path)
            
            # Store summary
            results_summary.append({
                'file': filename,
                'total_nodes': len(node_ids),
                'no_annotation': (df['annotation_type'] == 'no_annotation').sum(),
                'dimension': (df['annotation_type'] == 'dimension').sum(),
                'text': (df['annotation_type'] == 'text').sum(),
                'both': (df['annotation_type'] == 'both').sum()
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append({
                'file': filename,
                'error': str(e)
            })
    
    # Print overall summary
    print("\n" + "="*80)
    print("=== BATCH PROCESSING COMPLETE ===")
    print("="*80)
    print(f"\n{'File':<35} {'Nodes':<8} {'None':<8} {'Dim':<8} {'Text':<8} {'Both':<8}")
    print("-" * 80)
    
    for result in results_summary:
        if 'error' in result:
            print(f"{result['file']:<35} ERROR: {result['error']}")
        else:
            print(f"{result['file']:<35} "
                  f"{result['total_nodes']:<8} "
                  f"{result['no_annotation']:<8} "
                  f"{result['dimension']:<8} "
                  f"{result['text']:<8} "
                  f"{result['both']:<8}")
    
    # Calculate totals
    total_nodes = sum(r.get('total_nodes', 0) for r in results_summary)
    total_none = sum(r.get('no_annotation', 0) for r in results_summary)
    total_dim = sum(r.get('dimension', 0) for r in results_summary)
    total_text = sum(r.get('text', 0) for r in results_summary)
    total_both = sum(r.get('both', 0) for r in results_summary)
    
    print("-" * 80)
    print(f"{'TOTAL':<35} "
          f"{total_nodes:<8} "
          f"{total_none:<8} "
          f"{total_dim:<8} "
          f"{total_text:<8} "
          f"{total_both:<8}")
    
    # Print percentages
    if total_nodes > 0:
        print("\nPercentages:")
        print(f"  No annotation:     {100*total_none/total_nodes:5.1f}%")
        print(f"  Dimension:         {100*total_dim/total_nodes:5.1f}%")
        print(f"  Text:              {100*total_text/total_nodes:5.1f}%")
        print(f"  Both:              {100*total_both/total_nodes:5.1f}%")
    
    print(f"\n✓ All predictions saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
