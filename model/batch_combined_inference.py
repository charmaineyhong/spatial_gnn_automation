#!/usr/bin/env python3
"""
Batch Combined Inference: Process all GraphML files in a directory

Runs combined dimension + text inference on all GraphML files.
"""

import os
import argparse
import glob
from combined_inference import (
    parse_graphml_to_pyg,
    load_models,
    predict_combined,
    save_predictions
)
import torch


def main():
    parser = argparse.ArgumentParser(description='Batch combined inference on all GraphML files')
    parser.add_argument('--input_dir', default='./3.InputML',
                       help='Directory containing GraphML files')
    parser.add_argument('--output_dir', default='./6.CombinedPredictions',
                       help='Output directory for predictions')
    parser.add_argument('--dim_model', default='./5.OutputML_GAT_DIM/trained_model.pth',
                       help='Path to dimension model')
    parser.add_argument('--text_model', 
                       default='./5.OutputML_GAT_TEXT/trained_model_text.pth',
                       help='Path to text model')
    parser.add_argument('--dim_threshold', type=float, default=0.4,
                       help='Threshold for dimension prediction')
    parser.add_argument('--text_threshold', type=float, default=0.35,
                       help='Threshold for text prediction')
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
    
    print(f"=== Batch Combined Inference ===")
    print(f"Found {len(graphml_files)} GraphML files")
    print(f"Dimension model: {args.dim_model}")
    print(f"Text model: {args.text_model}")
    print(f"Device: {args.device}")
    print(f"Thresholds: dim={args.dim_threshold}, text={args.text_threshold}")
    print(f"Output directory: {args.output_dir}")
    
    # Load models once
    print("\nLoading models...")
    # Parse first file to get input_dim
    data_temp, _ = parse_graphml_to_pyg(graphml_files[0])
    input_dim = data_temp.x.shape[1]
    
    device = torch.device(args.device)
    dim_model, text_model = load_models(args.dim_model, args.text_model, input_dim, device)
    print("Models loaded successfully\n")
    
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
            predictions, probabilities = predict_combined(
                dim_model, text_model, data, device,
                args.dim_threshold, args.text_threshold
            )
            
            # Save predictions
            df = save_predictions(node_ids, predictions, probabilities, output_path)
            
            # Store summary
            results_summary.append({
                'file': filename,
                'total_nodes': len(node_ids),
                'no_annotation': (df['annotation_type'] == 'none').sum(),
                'dimension_only': (df['annotation_type'] == 'dimension_only').sum(),
                'text_only': (df['annotation_type'] == 'text_only').sum(),
                'both': (df['annotation_type'] == 'both').sum()
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append({
                'file': filename,
                'error': str(e)
            })
    
    # Print overall summary
    print("\n" + "="*60)
    print("=== BATCH PROCESSING COMPLETE ===")
    print("="*60)
    print(f"\n{'File':<30} {'Nodes':<8} {'None':<8} {'Dim':<8} {'Text':<8} {'Both':<8}")
    print("-" * 70)
    
    for result in results_summary:
        if 'error' in result:
            print(f"{result['file']:<30} ERROR: {result['error']}")
        else:
            print(f"{result['file']:<30} "
                  f"{result['total_nodes']:<8} "
                  f"{result['no_annotation']:<8} "
                  f"{result['dimension_only']:<8} "
                  f"{result['text_only']:<8} "
                  f"{result['both']:<8}")
    
    # Calculate totals
    total_nodes = sum(r.get('total_nodes', 0) for r in results_summary)
    total_none = sum(r.get('no_annotation', 0) for r in results_summary)
    total_dim = sum(r.get('dimension_only', 0) for r in results_summary)
    total_text = sum(r.get('text_only', 0) for r in results_summary)
    total_both = sum(r.get('both', 0) for r in results_summary)
    
    print("-" * 70)
    print(f"{'TOTAL':<30} "
          f"{total_nodes:<8} "
          f"{total_none:<8} "
          f"{total_dim:<8} "
          f"{total_text:<8} "
          f"{total_both:<8}")
    
    print(f"\n✓ All predictions saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
