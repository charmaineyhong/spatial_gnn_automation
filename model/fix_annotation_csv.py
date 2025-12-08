#!/usr/bin/env python3
"""
Fix malformed annotation.csv files by merging broken multi-line text values.
"""

import os
import csv

def fix_annotation_csv(file_path):
    """
    Fix annotation CSV by merging broken lines back into single rows.
    
    Strategy:
    - Read file line by line
    - If a line has fewer columns than expected, it's part of a multi-line text value
    - Merge it with the previous line by replacing newline with a space
    """
    
    try:
        # Backup the original file
        backup_path = file_path + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"[BACKUP] Created {backup_path}")
        
        # Read all lines
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            print(f"[SKIP] File too short: {file_path}")
            return False
        
        # Get header and expected column count
        header = lines[0].strip()
        expected_cols = header.count(',') + 1
        
        # Merge broken lines
        fixed_rows = [header]
        current_row = ""
        
        for i in range(1, len(lines)):
            line = lines[i].rstrip('\n')
            
            if not line.strip():
                continue
            
            # If we have a pending row, check if we need to merge
            if current_row:
                # Count columns in current accumulated row
                test_row = current_row + ' ' + line
                col_count = test_row.count(',') + 1
                
                if col_count <= expected_cols:
                    # Still need more content, merge and continue
                    current_row = test_row
                    if col_count == expected_cols:
                        # Complete row now
                        fixed_rows.append(current_row)
                        current_row = ""
                else:
                    # Previous row was complete, save it and start new row
                    fixed_rows.append(current_row)
                    current_row = line
            else:
                # Start new row
                col_count = line.count(',') + 1
                if col_count == expected_cols:
                    # Complete row
                    fixed_rows.append(line)
                else:
                    # Incomplete row, accumulate
                    current_row = line
        
        # Don't forget the last row if pending
        if current_row:
            fixed_rows.append(current_row)
        
        # Write fixed content
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            for row in fixed_rows:
                f.write(row + '\n')
        
        merged_count = len(lines) - 1 - (len(fixed_rows) - 1)
        print(f"[FIXED] {file_path} - merged {merged_count} broken lines, final: {len(fixed_rows)-1} rows")
        return True
        
    except Exception as e:
        print(f"[ERROR] Could not fix {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Fix all annotation.csv files
    extracted_data_base = "../Extracted Data"
    
    if not os.path.exists(extracted_data_base):
        print(f"[ERROR] Directory not found: {extracted_data_base}")
        return
    
    folders = [f for f in os.listdir(extracted_data_base) 
               if os.path.isdir(os.path.join(extracted_data_base, f))]
    
    print(f"Fixing annotation.csv files in {len(folders)} PPVC folders...\n")
    
    fixed_count = 0
    for folder in sorted(folders):
        annotation_path = os.path.join(extracted_data_base, folder, "annotation.csv")
        
        if os.path.exists(annotation_path):
            print(f"Processing {folder}...")
            if fix_annotation_csv(annotation_path):
                fixed_count += 1
            print()
    
    print(f"\n[DONE] Fixed {fixed_count} annotation.csv files")
    print("\nNow run:")
    print("  python3 csv_to_graphml.py")
    print("  python3 map_text_to_elements.py")

if __name__ == "__main__":
    main()
