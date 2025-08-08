#!/usr/bin/env python3
"""
Create ground truth files from LangExtract classification results
"""

import json
import os
import glob

def create_ground_truth_files():
    """Create ground truth files from classification results"""
    
    # Create ground_truth directory
    os.makedirs('ground_truth', exist_ok=True)
    
    # Find all classification files
    classification_files = glob.glob('ocr_results/*_classification.json')
    
    print(f"🎯 Creating ground truth files from {len(classification_files)} classifications...")
    
    for classification_file in classification_files:
        # Get base filename without _classification.json
        base_name = os.path.basename(classification_file).replace('_classification.json', '')
        
        # Load classification result
        with open(classification_file, 'r') as f:
            classification = json.load(f)
        
        # Create ground truth content
        ground_truth_content = f"Category: {classification['category']}\n"
        
        if classification['tags']:
            ground_truth_content += "Tags: " + ", ".join(classification['tags']) + "\n"
        else:
            ground_truth_content += "Tags: None\n"
        
        # Save ground truth file
        gt_filename = f"ground_truth/{base_name}_gt.txt"
        with open(gt_filename, 'w') as f:
            f.write(ground_truth_content)
        
        print(f"  ✅ {gt_filename}")
    
    print(f"\n✅ Created {len(classification_files)} ground truth files!")

if __name__ == "__main__":
    create_ground_truth_files()
