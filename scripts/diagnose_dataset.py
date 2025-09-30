#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose ExoGlove dataset issues
"""

import os
import glob
from pathlib import Path
import numpy as np

def diagnose_dataset():
    print("Diagnosing ExoGlove dataset...")
    
    # Check class distribution
    class_counts = {}
    total_annotations = 0
    invalid_files = []
    
    for split in ['train', 'valid', 'test']:
        label_dir = Path(split) / "labels"
        if not label_dir.exists():
            continue
            
        print(f"\nChecking {split} split...")
        
        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    invalid_files.append(f"{label_file}: empty file")
                    continue
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        invalid_files.append(f"{label_file}:{line_num} - wrong format")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        if class_id < 0:
                            invalid_files.append(f"{label_file}:{line_num} - negative class ID: {class_id}")
                            continue
                        if class_id > 8:
                            invalid_files.append(f"{label_file}:{line_num} - class ID too high: {class_id}")
                            continue
                        
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_annotations += 1
                        
                        # Check coordinates
                        for i, coord in enumerate(parts[1:], 1):
                            coord_val = float(coord)
                            if coord_val < 0 or coord_val > 1:
                                invalid_files.append(f"{label_file}:{line_num} - invalid coord {i}: {coord_val}")
                                break
                                
                    except ValueError as e:
                        invalid_files.append(f"{label_file}:{line_num} - parse error: {e}")
                        continue
                        
            except Exception as e:
                invalid_files.append(f"{label_file} - file error: {e}")
    
    print(f"\nDataset Summary:")
    print(f"Total annotations: {total_annotations}")
    print(f"Invalid files/annotations: {len(invalid_files)}")
    
    print(f"\nClass distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
        print(f"  Class {class_id}: {count} annotations ({percentage:.1f}%)")
    
    # Check for missing classes
    expected_classes = set(range(9))  # 0-8
    found_classes = set(class_counts.keys())
    missing_classes = expected_classes - found_classes
    
    if missing_classes:
        print(f"\nMissing classes: {sorted(missing_classes)}")
    
    # Show first few invalid files
    if invalid_files:
        print(f"\nFirst 10 invalid files/annotations:")
        for invalid in invalid_files[:10]:
            print(f"  - {invalid}")
        if len(invalid_files) > 10:
            print(f"  ... and {len(invalid_files) - 10} more")
    
    return len(invalid_files) == 0, class_counts

def fix_class_issues():
    """Fix class ID issues by ensuring all classes 0-8 are present"""
    print("\nFixing class issues...")
    
    # Check if we need to add missing classes
    _, class_counts = diagnose_dataset()
    
    # If we have classes but not all 0-8, we might need to adjust
    if class_counts and max(class_counts.keys()) > 8:
        print("Found class IDs > 8, this might be the issue")
        return False
    
    return True

if __name__ == "__main__":
    is_valid, class_counts = diagnose_dataset()
    
    if is_valid:
        print("\nDataset appears to be valid!")
    else:
        print("\nDataset has issues that need fixing.")
        fix_class_issues()
