import pandas as pd

new = pd.read_csv('C:/Users/charm/Downloads/realppvc/annotation_with_targets.csv')
train = pd.read_csv('C:/Users/charm/Spatial GNN/Extracted Data/PPVC 13_Typ/annotation_with_targets.csv')

print("="*60)
print("COMPARISON: NEW vs TRAINING ANNOTATION_WITH_TARGETS")
print("="*60)

print("\nNEW (realppvc):")
print(f"  Rows: {len(new)}")
print(f"  Columns: {list(new.columns)}")

print("\nTRAINING (PPVC 13):")
print(f"  Rows: {len(train)}")
print(f"  Columns: {list(train.columns)}")

print(f"\nDIFFERENCE: Training has {len(train) - len(new)} MORE rows")

print("\n" + "="*60)
print("CATEGORY DISTRIBUTION")
print("="*60)

if 'category' in new.columns:
    print("\nNEW:")
    print(new['category'].value_counts())
else:
    print("\nNEW: No category column found!")

print("\nTRAINING:")
print(train['category'].value_counts())

print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)

print("\nFirst 5 rows from NEW:")
print(new.head())

print("\nFirst 5 rows from TRAINING:")
print(train.head())

print("\n" + "="*60)
print("POSSIBLE ISSUES")
print("="*60)

if len(new) < len(train):
    print(f"1. NEW export is missing {len(train) - len(new)} annotations")
    print("   → Check if whitelist filtering is working in C# export")
    
if list(new.columns) != list(train.columns):
    print("2. Column structure is different")
    print(f"   NEW columns: {list(new.columns)}")
    print(f"   TRAINING columns: {list(train.columns)}")

if 'category' in new.columns and 'category' in train.columns:
    new_cats = set(new['category'].unique())
    train_cats = set(train['category'].unique())
    if new_cats != train_cats:
        print("3. Different categories found:")
        print(f"   Only in NEW: {new_cats - train_cats}")
        print(f"   Only in TRAINING: {train_cats - new_cats}")
