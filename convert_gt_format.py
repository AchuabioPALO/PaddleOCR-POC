#!/usr/bin/env python3
"""
Convert text ground truth files to JSON format for evaluation
"""

import json
import os
from pathlib import Path

def convert_txt_to_json_gt():
    """Convert .txt ground truth files to .json format"""
    
    print("🔄 Converting ground truth files from .txt to .json format...")
    
    gt_dir = Path("ground_truth")
    txt_files = list(gt_dir.glob("*_gt.txt"))
    
    print(f"Found {len(txt_files)} .txt ground truth files")
    
    for txt_file in txt_files:
        print(f"Converting: {txt_file.name}")
        
        # Read the .txt file
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Parse the content
        lines = content.split('\n')
        category = ""
        tags = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("Category:"):
                category = line.replace("Category:", "").strip()
            elif line.startswith("Tags:"):
                tags_str = line.replace("Tags:", "").strip()
                if tags_str:
                    tags = [tag.strip() for tag in tags_str.split(',')]
        
        # Create JSON ground truth
        gt_data = {
            "category": category,
            "tags": tags,
            "confidence": 1.0,
            "keywords_found": [],
            "detailed_tags": [
                {
                    "tag": tag,
                    "confidence": 1.0,
                    "method": "Manual Ground Truth"
                } for tag in tags
            ],
            "method": "Manual Ground Truth"
        }
        
        # Write JSON file with proper formatting
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, indent=2, ensure_ascii=False, separators=(',', ': '))
        
        # Save as JSON
        json_file = txt_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(gt_json, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Saved: {json_file.name}")
        print(f"     Category: {category}")
        print(f"     Tags: {tags}")
        print()
    
    print(f"🎯 Converted {len(txt_files)} ground truth files!")

if __name__ == "__main__":
    convert_txt_to_json_gt()
