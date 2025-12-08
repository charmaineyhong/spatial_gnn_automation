#!/usr/bin/env python3
"""
inspect_annotation.py

Quick data inspection for PPVC annotation.csv files.

For each annotation.csv under "../Extracted Data", this script prints:

- DIMENSION rows        : how many rows look like dimensions
- TEXT_UMBRELLA rows    : how many rows are "real" text notes we care about
- Skipped (other stuff) : everything else

TEXT_UMBRELLA is defined *generically* (no manual room/struct keyword lists):

  - category == "TextNotes"   (case-insensitive)
  - annotation_type == "text_note"
  - value is non-empty
  - value contains at least one alphabetic character
  - value does NOT contain obvious junk like "Detail Filled Region" or "Opening"

This is only for analysis / imbalance checking. It does NOT change training labels.
"""

import os
import pandas as pd
import math


BASE_DIR = "../Extracted Data"


# ----------------------------
# Small helpers
# ----------------------------

def is_nan_like(v):
    """Return True if v is None / NaN / empty / 'nan' / 'null' text."""
    if v is None:
        return True
    # Real NaN (float)
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:
        pass
    s = str(v).strip()
    if s == "":
        return True
    if s.lower() in {"nan", "none", "null"}:
        return True
    return False


def is_dimension_row(category, annotation_type):
    """
    Decide if a row is a DIMENSION.
    Based on broad patterns in category / annotation_type.
    """
    cat_u = str(category).upper()
    atype_u = str(annotation_type).upper()

    if "DIMENSION" in cat_u or "DIMENSIONS" in cat_u:
        return True
    if "DIM" in atype_u:  # catches 'Linear Dimension Style', etc.
        return True
    return False


def is_text_umbrella_row(category, annotation_type, value):
    """
    Decide if a row belongs to TEXT_UMBRELLA:
    - category == "TextNotes"
    - annotation_type == "text_note"
    - non-empty value
    - value has at least one alphabetic character
    - exclude known junk like 'Detail Filled Region', 'Opening'
    """
    cat = str(category).strip().lower()
    atype = str(annotation_type).strip().lower()

    if not (cat == "textnotes" and atype == "text_note"):
        return False

    if is_nan_like(value):
        return False

    val = str(value).strip()
    if not val:
        return False

    # Require at least one alphabetic character (so we skip pure numeric like "73.86")
    if not any(ch.isalpha() for ch in val):
        return False

    # Exclude obvious non-label junk (you can tweak this set if needed)
    val_u = val.upper()
    EXCLUDE = {
        "DETAIL FILLED REGION",
        "OPENING",
    }
    for ex in EXCLUDE:
        if ex in val_u:
            return False

    return True


# ----------------------------
# Main inspection logic
# ----------------------------

def main():
    # Find all annotation_with_targets.csv under BASE_DIR, fallback to annotation.csv
    annotation_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        # Prefer annotation_with_targets.csv
        if "annotation_with_targets.csv" in files:
            annotation_files.append(os.path.join(root, "annotation_with_targets.csv"))
        elif "annotation.csv" in files:
            annotation_files.append(os.path.join(root, "annotation.csv"))

    if not annotation_files:
        print(f"No annotation.csv found under {BASE_DIR}")
        return

    total_dim = 0
    total_text = 0
    total_skipped = 0

    print("Found annotation files:")
    for p in annotation_files:
        print("  -", p)
    print("\n==== PER-FILE SUMMARY ====\n")

    for csv_path in sorted(annotation_files):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Could not read {csv_path}: {e}")
            continue

        n_dim = 0
        n_text = 0
        text_values_sample = []
        dim_values_sample = []

        for _, row in df.iterrows():
            cat = row.get("category", "")
            atype = row.get("annotation_type", "")
            value = row.get("value", "")

            if is_dimension_row(cat, atype):
                n_dim += 1
                if not is_nan_like(value) and len(dim_values_sample) < 10:
                    dim_values_sample.append(str(value))
                continue

            if is_text_umbrella_row(cat, atype, value):
                n_text += 1
                if not is_nan_like(value) and len(text_values_sample) < 10:
                    text_values_sample.append(str(value))
                continue

        n_total = len(df)
        n_skip = n_total - n_dim - n_text

        total_dim += n_dim
        total_text += n_text
        total_skipped += n_skip

        print(f"File: {csv_path}")
        print(f"  DIMENSION rows     : {n_dim}")
        print(f"  TEXT_UMBRELLA rows : {n_text}")
        print(f"  Skipped (other)    : {n_skip}")
        if dim_values_sample:
            print(f"  Sample DIM 'value' entries  : {dim_values_sample}")
        if text_values_sample:
            print(f"  Sample TEXT 'value' entries : {text_values_sample}")
        print("")

    print("==== TOTAL ACROSS ALL FILES ====")
    print(f"DIMENSION rows        : {total_dim}")
    print(f"TEXT_UMBRELLA rows    : {total_text}")
    print(f"Skipped (other stuff) : {total_skipped}")


if __name__ == "__main__":
    main()
