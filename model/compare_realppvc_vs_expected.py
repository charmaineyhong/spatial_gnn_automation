"""
Compare realppvc predictions.csv with PPVC 14 expected predictions.
Shows which elements have different predictions.
"""

import pandas as pd

# Load both CSVs
realppvc_path = r"c:\Users\charm\Downloads\realppvc\predictions.csv"
expected_path = r"c:\Users\charm\Spatial GNN\model\ppvc14_expected_predictions.csv"

realppvc_df = pd.read_csv(realppvc_path)
expected_df = pd.read_csv(expected_path)

print("="*70)
print("Comparing RealPPVC predictions vs PPVC 14 Expected predictions")
print("="*70)

# Rename columns for consistency
realppvc_df = realppvc_df.rename(columns={'node_id': 'element_id', 'predicted_class': 'predicted_label'})

# Merge on element_id
merged = expected_df.merge(
    realppvc_df[['element_id', 'predicted_label', 'confidence', 'annotation_type']], 
    on='element_id', 
    suffixes=('_expected', '_realppvc'),
    how='outer'
)

# Find differences
different = merged[merged['predicted_label_expected'] != merged['predicted_label_realppvc']].copy()

print(f"\nTotal elements in expected: {len(expected_df)}")
print(f"Total elements in realppvc: {len(realppvc_df)}")
print(f"Total differences: {len(different)}")

if len(different) > 0:
    print(f"\n{'='*70}")
    print("Elements with different predictions:")
    print(f"{'='*70}")
    
    label_names = {0: 'no_annotation', 1: 'dimension', 2: 'text', 3: 'both'}
    
    # Add readable labels
    different['expected_label_name'] = different['predicted_label_expected'].map(label_names)
    different['realppvc_label_name'] = different['predicted_label_realppvc'].map(label_names)
    
    # Sort by element_id
    different = different.sort_values('element_id')
    
    # Display with formatting
    print(f"\n{'Element ID':<12} {'Expected':<15} {'RealPPVC':<15} {'Confidence':<12} {'Match?'}")
    print("-"*70)
    
    for _, row in different.iterrows():
        expected_label = f"{int(row['predicted_label_expected'])} ({row['expected_label_name']})"
        realppvc_label = f"{int(row['predicted_label_realppvc'])} ({row['realppvc_label_name']})"
        confidence = f"{row['confidence']:.4f}"
        match = "✓" if row['predicted_label_expected'] == row['predicted_label_realppvc'] else "✗"
        
        print(f"{int(row['element_id']):<12} {expected_label:<15} {realppvc_label:<15} {confidence:<12} {match}")
    
    # Summary by change type
    print(f"\n{'='*70}")
    print("Change Summary:")
    print(f"{'='*70}")
    
    change_counts = {}
    for _, row in different.iterrows():
        exp = row['expected_label_name']
        real = row['realppvc_label_name']
        change_type = f"{exp} → {real}"
        change_counts[change_type] = change_counts.get(change_type, 0) + 1
    
    for change_type, count in sorted(change_counts.items(), key=lambda x: -x[1]):
        print(f"  {change_type:<40}: {count:3d}")
    
    # Save to CSV
    output_path = "realppvc_vs_ppvc14_differences.csv"
    different.to_csv(output_path, index=False)
    print(f"\nFull comparison saved to: {output_path}")
    
else:
    print("\n✓ All predictions match perfectly!")

# Check for elements only in one file
only_in_expected = set(expected_df['element_id']) - set(realppvc_df['element_id'])
only_in_realppvc = set(realppvc_df['element_id']) - set(expected_df['element_id'])

if only_in_expected:
    print(f"\n⚠ Elements only in expected (not in realppvc): {len(only_in_expected)}")
    print(f"  {sorted(only_in_expected)[:10]}...")

if only_in_realppvc:
    print(f"\n⚠ Elements only in realppvc (not in expected): {len(only_in_realppvc)}")
    print(f"  {sorted(only_in_realppvc)[:10]}...")
