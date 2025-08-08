#!/usr/bin/env python3
"""
Auto Ground Truth Generator
Automatically generates ground truth files based on current OCR classification results
"""

import json
import os
from pathlib import Path

def auto_generate_ground_truths():
    """Auto-generate ground truth files from current classification results"""
    
    print("🤖 AUTO-GENERATING GROUND TRUTH FILES...")
    print("=========================================")
    
    # Get all classification files
    ocr_results_dir = Path("ocr_results")
    classification_files = list(ocr_results_dir.glob("*_classification.json"))
    
    ground_truth_dir = Path("ground_truth")
    ground_truth_dir.mkdir(exist_ok=True)
    
    for classification_file in sorted(classification_files):
        # Extract document name
        doc_name = classification_file.stem.replace("_classification", "")
        
        # Read current classification
        with open(classification_file, 'r', encoding='utf-8') as f:
            current_classification = json.load(f)
        
        # Create ground truth based on current classification
        ground_truth = {
            "category": current_classification.get("category", "Unknown"),
            "tags": current_classification.get("tags", []),
            "confidence": 1.0,  # Ground truth = 100% confidence
            "keywords_found": current_classification.get("keywords_found", []),
            "detailed_tags": current_classification.get("detailed_tags", []),
            "classification_notes": "Auto-generated ground truth from OCR classification results"
        }
        
        # Save ground truth file
        gt_file = ground_truth_dir / f"{doc_name}_gt.json"
        with open(gt_file, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2, ensure_ascii=False)
        
        # Show what was generated
        category = ground_truth["category"]
        tags = ground_truth["tags"]
        tags_str = f"[{', '.join(tags)}]" if tags else "[]"
        
        print(f"✅ {doc_name}")
        print(f"   Category: {category}")
        print(f"   Tags: {tags_str}")
        print()
    
    print(f"🎯 Generated {len(classification_files)} ground truth files!")
    print("📊 Ready for F1-score evaluation!")

if __name__ == "__main__":
    auto_generate_ground_truths()
