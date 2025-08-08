#!/usr/bin/env python3
"""
Ground Truth Template Creator
Creates ground truth templates for F1-score evaluation
"""

import json
import os
from pathlib import Path

def create_ground_truth_templates():
    """Create ground truth templates based on processed OCR results"""
    
    # Available categories (from folder structure)
    categories = [
        "Agreement", "Cleaning", "Commissioning Record", "Customer Complaint",
        "Decommissioning Notice", "Defect Notification to Customer", "HEC Info",
        "Inspection", "Letter", "Replacement Notice", "Slope Work", "SRIC", "SS Maintenance"
    ]
    
    # Available tags (exactly 8 as specified)
    tags = [
        "Agreement (Access)",
        "Agreement (Fire)",
        "Agreement (Ventilation)",
        "Defect (Access)",
        "Defect (Civil)",
        "Defect (Fire)",
        "Defect (Ventilation)",
        "VSC/VIC"
    ]
    
    # Template structure
    template = {
        "category": "",  # To be filled manually
        "tags": [],     # To be filled manually (can be multiple)
        "confidence": 1.0,  # Manual annotation = 100% confidence
        "keywords_found": [],  # Manual annotation doesn't need keywords
        "detailed_tags": [],   # Will be populated automatically
        "classification_notes": "Manual ground truth annotation",
        "instructions": {
            "category": f"Choose ONE from: {categories}",
            "tags": f"Choose MULTIPLE from: {tags}",
            "note": "Remove this instructions field when done editing"
        }
    }
    
    # Get list of processed documents
    ocr_results_dir = Path("ocr_results")
    classification_files = list(ocr_results_dir.glob("*_classification.json"))
    
    print(f"🏷️  Creating ground truth templates for {len(classification_files)} documents...")
    print("=" * 60)
    
    ground_truth_dir = Path("ground_truth")
    ground_truth_dir.mkdir(exist_ok=True)
    
    for classification_file in sorted(classification_files):
        # Extract document name
        doc_name = classification_file.stem.replace("_classification", "")
        
        # Read current classification for reference
        with open(classification_file, 'r', encoding='utf-8') as f:
            current_classification = json.load(f)
        
        # Create template with current classification as suggestion
        gt_template = template.copy()
        gt_template["suggested_category"] = current_classification.get("category", "")
        gt_template["suggested_tags"] = current_classification.get("tags", [])
        gt_template["auto_keywords_found"] = current_classification.get("keywords_found", [])
        
        # Save ground truth template
        gt_file = ground_truth_dir / f"{doc_name}_gt.json"
        with open(gt_file, 'w', encoding='utf-8') as f:
            json.dump(gt_template, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created: {gt_file.name}")
        print(f"   Current: {current_classification.get('category', 'Unknown')} | {current_classification.get('tags', [])}")
        print()
    
    print(f"🎯 Ground truth templates created! Edit the files in ground_truth/ folder:")
    print("   1. Set the correct 'category' field")
    print("   2. Set the correct 'tags' field (can be multiple)")
    print("   3. Remove the 'instructions' field when done")
    print("   4. Run F1-score evaluation when ready")

if __name__ == "__main__":
    create_ground_truth_templates()
