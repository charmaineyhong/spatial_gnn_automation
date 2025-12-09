import pandas as pd

# Load predictions from testing ppvc folder
pred = pd.read_csv('C:/Users/charm/Downloads/testing ppvc/predictions.csv')

# Load CLEANED ground truth (this is what the model was trained on)
truth = pd.read_csv('C:/Users/charm/Spatial GNN/Extracted Data/PPVC 13_Typ/annotation_with_targets.csv')

# Parse target_element_ids (pipe-separated list) and create one row per target element
truth_expanded = []
for _, row in truth.iterrows():
    targets = str(row['target_element_ids'])
    if pd.notna(targets) and targets != 'nan':
        for target_id in targets.split('|'):
            target_id = target_id.strip()
            if target_id:
                truth_expanded.append({
                    'element_id': int(target_id),
                    'category': row['category'],
                    'annotation_id': row['annotation_id']
                })

truth_df = pd.DataFrame(truth_expanded)

# Map category to class
def map_category(cat):
    cat = str(cat).lower()
    if 'dimension' in cat:
        return 'dimension'
    elif 'text' in cat:
        return 'text'
    else:
        return 'unknown'

truth_df['true_class'] = truth_df['category'].apply(map_category)

# Check for elements with both dimension and text
element_classes = truth_df.groupby('element_id')['true_class'].apply(
    lambda x: 'both' if len(set(x)) > 1 else x.iloc[0]
)

# Create final ground truth
truth_final = pd.DataFrame({
    'element_id': element_classes.index,
    'true_class': element_classes.values
})

# Merge predictions with ground truth
merged = pd.merge(pred, truth_final, left_on='node_id', right_on='element_id', how='inner')

# Overall accuracy
correct = (merged['predicted_class'] == merged['true_class']).sum()
total = len(merged)
print(f'Overall Accuracy: {correct}/{total} = {100*correct/total:.1f}%')
print(f'Matched {total} elements out of {len(pred)} predictions and {len(truth_final)} ground truth')

# Per-class accuracy
print(f'\nPer-class results (Recall):')
for cls in ['dimension', 'text', 'both']:
    cls_data = merged[merged['true_class'] == cls]
    cls_correct = (cls_data['predicted_class'] == cls_data['true_class']).sum()
    cls_total = len(cls_data)
    if cls_total > 0:
        print(f'  {cls:20s}: {cls_correct:3d}/{cls_total:3d} = {100*cls_correct/cls_total:5.1f}%')

# Show misclassified samples
print(f'\nMisclassified samples:')
wrong = merged[merged['predicted_class'] != merged['true_class']]
if len(wrong) > 0:
    print(wrong[['node_id', 'predicted_class', 'confidence', 'true_class']].head(10).to_string())
else:
    print('None!')
