#!/usr/bin/env python3
"""
Single PPVC combined inference from nodes.csv + edges.csv

Usage (example):

  python run_inference_from_csv.py ^
    --nodes_csv "C:/PPVC_AI/Test/nodes.csv" ^
    --edges_csv "C:/PPVC_AI/Test/edges.csv" ^
    --out_csv   "C:/PPVC_AI/Test/predictions.csv"
"""

import argparse
import torch
from combined_inference import (
    parse_csv_to_pyg,
    load_models,
    predict_combined,
    save_predictions,
)


def main():
    parser = argparse.ArgumentParser(description="Single-file combined inference from CSV")
    parser.add_argument("--nodes_csv", required=True, help="Path to nodes.csv")
    parser.add_argument("--edges_csv", required=True, help="Path to edges.csv")
    parser.add_argument("--out_csv", required=True, help="Output predictions.csv path")

    parser.add_argument(
        "--dim_model",
        default="./5.OutputML_GAT_DIM/trained_model.pth",
        help="Path to dimension model",
    )
    parser.add_argument(
        "--text_model",
        default="./5.OutputML_GAT_TEXT/trained_model_text.pth",
        help="Path to text model",
    )
    parser.add_argument("--dim_threshold", type=float, default=0.7)
    parser.add_argument("--text_threshold", type=float, default=0.4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)

    print("=== CSV GNN Inference ===")
    print("nodes_csv:", args.nodes_csv)
    print("edges_csv:", args.edges_csv)
    print("dim_model:", args.dim_model)
    print("text_model:", args.text_model)
    print("device:", device)
    print(f"thresholds: dim={args.dim_threshold}, text={args.text_threshold}")

    # 1) Build graph from CSV
    data = parse_csv_to_pyg(args.nodes_csv, args.edges_csv)
    node_ids = getattr(data, "node_ids", list(range(data.num_nodes)))
    input_dim = data.x.shape[1]
    print(f"Loaded {len(node_ids)} nodes with {input_dim} features\n")

    # 2) Load models
    dim_model, text_model = load_models(
        args.dim_model,
        args.text_model,
        input_dim,
        device,
    )
    print("Models loaded successfully.\n")

    # 3) Run combined prediction
    predictions, probabilities = predict_combined(
        dim_model,
        text_model,
        data,
        device,
        args.dim_threshold,
        args.text_threshold,
    )

    # 4) Save predictions in your usual format
    df = save_predictions(node_ids, predictions, probabilities, args.out_csv)

    print(f"\n Inference complete. Saved predictions to:\n  {args.out_csv}")
    print(df.head())


if __name__ == "__main__":
    main()
