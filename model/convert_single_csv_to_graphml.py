#!/usr/bin/env python3
"""
Convert a single nodes.csv + edges.csv pair into a GraphML file for inference
"""

import argparse
import pandas as pd
import networkx as nx


def csv_to_graphml(nodes_csv, edges_csv, output_graphml):
    """Convert CSV files to GraphML format matching training data structure"""
    
    # Read CSVs
    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Map category to element_type (matching training data)
    category_map = {
        "Walls": "Wall",
        "Structural Framing": "Beam",
        "Structural Foundation": "Foundation",
        "Floors": "Slab",
        "Generic Models": "GenericModel",
        "Rooms": "Zone",
        "Levels": "Level",
    }
    
    # Add nodes
    for _, row in df_nodes.iterrows():
        node_id = str(row['id'])
        category = str(row.get('category', 'Unknown'))
        
        # Map category to element_type
        element_type = category_map.get(category, "Unknown")
        
        # Create node attributes matching GraphML training format
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
        
        G.add_node(node_id, **node_attrs)
    
    # Add edges
    for _, row in df_edges.iterrows():
        src = str(row['src'])
        dst = str(row['dst'])
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst)
    
    # Save as GraphML
    nx.write_graphml(G, output_graphml)
    print(f"✓ Converted to GraphML: {output_graphml}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to GraphML for inference')
    parser.add_argument('--nodes_csv', required=True, help='Path to nodes.csv')
    parser.add_argument('--edges_csv', required=True, help='Path to edges.csv')
    parser.add_argument('--output', required=True, help='Path to output GraphML file')
    
    args = parser.parse_args()
    
    csv_to_graphml(args.nodes_csv, args.edges_csv, args.output)


if __name__ == '__main__':
    main()
