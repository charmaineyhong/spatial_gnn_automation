#!/usr/bin/env python3
"""
map_text_to_elements.py

This script solves the missing target_element_ids problem for TextNotes.

Problem:
- TextNotes rows in annotation.csv have empty target_element_ids
- The GAT training needs target_element_ids to label nodes as "Has Text"

Solution:
- Read annotation.csv and GraphML files
- For each TextNotes row, find the nearest element(s) using XY coordinates
- Write out a NEW annotation CSV with filled-in target_element_ids

Usage:
    python3 map_text_to_elements.py

Outputs:
    For each PPVC folder, creates:
        annotation_with_targets.csv  (augmented version with filled target_element_ids)
"""

import os
import sys
import math
import pandas as pd
import networkx as nx
from collections import defaultdict

# ================================
# Configuration
# ================================
EXTRACTED_DATA_BASE = "../Extracted Data"
GRAPHML_BASE = "./3.InputML"

# Distance threshold: if nearest element is farther than this, skip it
MAX_DISTANCE_THRESHOLD = 500.0  # Increased to capture more distant text annotations

# How many nearest elements to assign per text annotation
# Set to 1 for most conservative mapping, or 2-3 to label multiple nearby elements
NUM_NEAREST_ELEMENTS = 3  # Increased to label more elements per text

# ================================
# Mapping between PPVC names and folders
# ================================
PPVC_MAPPINGS = {
    "PPVC_01": {
        "graphml": "PPVC_01.graphml",
        "folders": ["PPVC 01, 20_Typ"]
    },
    "PPVC_02": {
        "graphml": "PPVC_02.graphml",
        "folders": ["PPVC 02_Typ 1"]
    },
    "PPVC_03": {
        "graphml": None,  # Not in 3.InputML, skip or add if available
        "folders": ["PPVC 03_Typ1_testing"]
    },
    "PPVC_04": {
        "graphml": "PPVC_04.graphml",
        "folders": ["PPVC 04_Typ 1"]
    },
    "PPVC_05": {
        "graphml": "PPVC_05.graphml",
        "folders": ["PPVC 05, 16_Typ"]
    },
    "PPVC_06": {
        "graphml": "PPVC_06.graphml",
        "folders": ["PPVC 06_Typ"]
    },
    "PPVC_21": {
        "graphml": "PPVC_21.graphml",
        "folders": ["PPVC 21,31_Typ"]
    },
    "PPVC_22": {
        "graphml": "PPVC_22.graphml",
        "folders": ["PPVC 22_Typ"]
    },
}


def is_text_annotation(row):
    """
    Determine if this annotation row should be treated as a text annotation
    that needs element mapping.
    
    Criteria:
    - category == "TextNotes" 
    - value is non-empty
    
    Optionally you can also include certain Tags if needed.
    """
    category = str(row.get("category", "")).strip()
    value = str(row.get("value", "")).strip()
    
    if category.upper() == "TEXTNOTES" and value != "":
        return True
    
    # If you want to include other tags (excluding structural framing / revision clouds):
    # if category.upper() == "TAGS":
    #     annotation_type = str(row.get("annotation_type", "")).strip().lower()
    #     if "structural framing" in annotation_type or "revision cloud" in annotation_type:
    #         return False
    #     if value != "":
    #         return True
    
    return False


def load_element_coordinates(graphml_path):
    """
    Load all elements from GraphML and extract their node_id and (pos_x, pos_y) coordinates.
    
    Returns:
        dict: { node_id (str): (x, y, element_type, view_info) }
    """
    if not os.path.exists(graphml_path):
        print(f"Warning: GraphML not found: {graphml_path}")
        return {}
    
    G = nx.read_graphml(graphml_path)
    elements = {}
    
    for node_id, data in G.nodes(data=True):
        # GraphML stores coordinates as pos_x, pos_y (based on key definitions)
        x = float(data.get("pos_x", 0.0))
        y = float(data.get("pos_y", 0.0))
        
        element_type = data.get("element_type", "unknown")
        
        # Store element info
        elements[str(node_id)] = {
            "x": x,
            "y": y,
            "element_type": element_type,
        }
    
    print(f"  Loaded {len(elements)} elements from {os.path.basename(graphml_path)}")
    return elements


def find_nearest_elements(text_x, text_y, elements, max_distance, num_nearest=1, debug=False):
    """
    Find the nearest element(s) to a text annotation point.
    
    Args:
        text_x, text_y: coordinates of the text annotation
        elements: dict of element_id -> {x, y, ...}
        max_distance: maximum distance to consider
        num_nearest: how many nearest elements to return
        debug: if True, print distance info
    
    Returns:
        list of element_ids (as strings), minimum distance
    """
    distances = []
    
    for elem_id, elem_data in elements.items():
        ex = elem_data["x"]
        ey = elem_data["y"]
        
        # Euclidean distance in 2D
        dist = math.sqrt((ex - text_x)**2 + (ey - text_y)**2)
        distances.append((dist, elem_id))
    
    if not distances:
        return [], float('inf')
    
    # Sort by distance
    distances.sort()
    
    min_dist = distances[0][0]
    
    if debug:
        print(f"      Text at ({text_x:.2f}, {text_y:.2f}) - Nearest: {min_dist:.2f}, Top 3: {distances[:3]}")
    
    # Filter by max distance and take top N
    result = []
    for dist, elem_id in distances[:num_nearest]:
        if dist <= max_distance:
            result.append(elem_id)
    
    return result, min_dist


def process_ppvc(ppvc_key, ppvc_config):
    """
    Process one PPVC:
    - Load GraphML
    - For each folder, load annotation.csv
    - Map text annotations to nearest elements
    - Write augmented annotation CSV
    """
    graphml_file = ppvc_config.get("graphml")
    if not graphml_file:
        print(f"\n[{ppvc_key}] No GraphML configured, skipping.")
        return
    
    graphml_path = os.path.join(GRAPHML_BASE, graphml_file)
    
    # Load element coordinates
    print(f"\n[{ppvc_key}] Loading elements from {graphml_file}...")
    elements = load_element_coordinates(graphml_path)
    
    if not elements:
        print(f"  No elements loaded, skipping {ppvc_key}.")
        return
    
    # Process each associated folder
    for folder_name in ppvc_config.get("folders", []):
        annotation_csv = os.path.join(EXTRACTED_DATA_BASE, folder_name, "annotation.csv")
        
        if not os.path.exists(annotation_csv):
            print(f"  Warning: annotation.csv not found at {annotation_csv}")
            continue
        
        print(f"\n  Processing {folder_name}/annotation.csv...")
        
        # The CSV has malformed rows where the value field contains unquoted newlines
        # Format: some rows are split across 2 lines:
        #   Line 1: annotation_id,view_name,view_type,category,annotation_type,value\n
        #   Line 2: ,x,y,extra,target_element_ids\n
        # We need to merge these continuation lines
        
        rows = []
        with open(annotation_csv, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            rows.append(header.split(','))
            
            buffer = []
            for line in f:
                line = line.rstrip('\n\r')
                parts = [p.strip() for p in line.split(',')]
                
                # If line starts with empty string (continuation line)
                if parts[0] == '' and buffer:
                    # Merge: buffer has first 6 cols, parts has last 4 (skip first empty)
                    # buffer: [id, view_name, view_type, category, annotation_type, value]
                    # parts:  ['', x, y, extra, target_element_ids] -> take [1:]
                    merged = buffer + parts[1:]
                    rows.append(merged)
                    buffer = []
                else:
                    # Check if this is a complete row (has all 10 fields)
                    if len(parts) >= 10:
                        rows.append(parts[:10])
                    else:
                        # Incomplete row, might continue on next line
                        buffer = parts
            
            # If there's a leftover buffer, add it (shouldn't happen in well-formed files)
            if buffer:
                # Pad with empty strings to reach 10 columns
                while len(buffer) < 10:
                    buffer.append('')
                rows.append(buffer[:10])
        
        # Convert to DataFrame
        df = pd.DataFrame(rows[1:], columns=rows[0])
        
        # Clean up the value column - remove embedded newlines/carriage returns
        if 'value' in df.columns:
            df['value'] = df['value'].astype(str).str.replace('\r', ' ').str.replace('\n', ' ').str.strip()
        
        # Convert numeric columns
        for col in ['x', 'y']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Track statistics
        text_rows = 0
        mapped_rows = 0
        min_distances = []
        
        # Process each row (debug first 5)
        for idx, row in df.iterrows():
            if is_text_annotation(row):
                text_rows += 1
                
                # Get text coordinates
                try:
                    text_x = float(row["x"])
                    text_y = float(row["y"])
                except (ValueError, KeyError):
                    print(f"    Row {idx}: invalid x/y coordinates, skipping")
                    continue
                
                # Find nearest element(s)
                debug_this = (text_rows <= 5)  # Debug first 5 text annotations
                nearest, min_dist = find_nearest_elements(
                    text_x, text_y, elements, 
                    MAX_DISTANCE_THRESHOLD, 
                    NUM_NEAREST_ELEMENTS,
                    debug=debug_this
                )
                
                min_distances.append(min_dist)
                
                if debug_this:
                    print(f"        -> Threshold: {MAX_DISTANCE_THRESHOLD}, Nearest: {nearest}, Mapped: {len(nearest) > 0}")
                
                if nearest:
                    # Fill in target_element_ids with pipe-separated list
                    df.at[idx, "target_element_ids"] = "|".join(nearest)
                    mapped_rows += 1
        
        # Write augmented CSV
        output_csv = os.path.join(EXTRACTED_DATA_BASE, folder_name, "annotation_with_targets.csv")
        df.to_csv(output_csv, index=False)
        
        if min_distances:
            avg_dist = sum(min_distances) / len(min_distances)
            min_of_mins = min(min_distances)
            max_of_mins = max(min_distances)
            print(f"    ✓ Text rows: {text_rows}, Mapped: {mapped_rows}")
            print(f"    ✓ Distance stats - Min: {min_of_mins:.2f}, Max: {max_of_mins:.2f}, Avg: {avg_dist:.2f}")
        else:
            print(f"    ✓ Text rows: {text_rows}, Mapped: {mapped_rows}")
        print(f"    ✓ Written to: {output_csv}")
        output_csv = os.path.join(EXTRACTED_DATA_BASE, folder_name, "annotation_with_targets.csv")
        df.to_csv(output_csv, index=False)
        
        print(f"    ✓ Text rows: {text_rows}, Mapped: {mapped_rows}")
        print(f"    ✓ Written to: {output_csv}")


def main():
    print("="*60)
    print("Text-to-Element Mapping Script")
    print("="*60)
    print(f"Max distance threshold: {MAX_DISTANCE_THRESHOLD}")
    print(f"Nearest elements per text: {NUM_NEAREST_ELEMENTS}")
    print()
    
    for ppvc_key in sorted(PPVC_MAPPINGS.keys()):
        process_ppvc(ppvc_key, PPVC_MAPPINGS[ppvc_key])
    
    print("\n" + "="*60)
    print("✓ All done!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated *_with_targets.csv files")
    print("2. Update your GAT training script to use 'annotation_with_targets.csv'")
    print("3. Re-run training to get proper 'Has Text' labels")


if __name__ == "__main__":
    main()