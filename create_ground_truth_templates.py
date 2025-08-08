#!/usr/bin/env python3
"""
Ground Truth Template Generator

This script creates ground truth template files based on OCR classification results
so you can manually edit them to be actual ground truth for F1-score evaluation.
"""

import json
import os
from pathlib import Path

def create_ground_truth_templates():
    """Create ground truth template files from OCR classification results."""
    
    ocr_results_dir = Path("ocr_results")
    ground_truth_dir = Path("ground_truth")
    
    # Create ground truth directory if it doesn't exist
    ground_truth_dir.mkdir(exist_ok=True)
    
    # Find all classification files
    classification_files = list(ocr_results_dir.glob("*_classification.json"))
    
    if not classification_files:
        print("No classification files found in ocr_results/")
        return
    
    print(f"Found {len(classification_files)} classification files")
    print("Creating ground truth templates...")
    print("=" * 60)
    
    ground_truth_data = {}
    
    for class_file in classification_files:
        try:
            # Load classification result
            with open(class_file, 'r', encoding='utf-8') as f:
                classification = json.load(f)
            
            # Extract PDF filename
            pdf_filename = class_file.name.replace('_classification.json', '.pdf')
            
            # Create ground truth template entry
            ground_truth_data[pdf_filename] = {
                "category": classification.get("category", "Unknown"),
                "tags": classification.get("tags", []),
                "confidence": classification.get("confidence", 0.0),
                "auto_generated": True,
                "needs_manual_review": True,
                "notes": f"Auto-generated from OCR classification. Please review and correct manually."
            }
            
            print(f"✓ {pdf_filename}")
            print(f"  Category: {classification.get('category', 'Unknown')}")
            print(f"  Tags: {', '.join(classification.get('tags', [])) or 'None'}")
            print(f"  Confidence: {classification.get('confidence', 0.0):.3f}")
            print()
            
        except Exception as e:
            print(f"✗ Error processing {class_file}: {str(e)}")
    
    # Save ground truth template as JSON
    gt_json_file = ground_truth_dir / "ground_truth_template.json"
    with open(gt_json_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth_data, f, ensure_ascii=False, indent=2)
    
    # Also save as CSV for easier editing
    import csv
    gt_csv_file = ground_truth_dir / "ground_truth_template.csv"
    with open(gt_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'category', 'tags', 'confidence', 'needs_review', 'notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for filename, data in ground_truth_data.items():
            writer.writerow({
                'filename': filename,
                'category': data['category'],
                'tags': '; '.join(data['tags']),
                'confidence': f"{data['confidence']:.3f}",
                'needs_review': 'YES',
                'notes': 'Auto-generated - please review and correct'
            })
    
    print("=" * 60)
    print("Ground truth templates created:")
    print(f"  JSON format: {gt_json_file}")
    print(f"  CSV format:  {gt_csv_file}")
    print()
    print("Next steps:")
    print("1. Review the CSV file and correct categories/tags manually")
    print("2. Save the corrected version as 'ground_truth_final.csv'")
    print("3. Run F1-score evaluation with the corrected ground truth")
    
    return ground_truth_data

def clean_folders():
    """Clean up and organize OCR results and ground truth folders."""
    
    print("🧹 CLEANING UP FOLDERS")
    print("=" * 60)
    
    ocr_dir = Path("ocr_results")
    gt_dir = Path("ground_truth")
    
    # Count current files
    all_files = list(ocr_dir.glob("*")) if ocr_dir.exists() else []
    classification_files = list(ocr_dir.glob("*_classification.json")) if ocr_dir.exists() else []
    detailed_files = list(ocr_dir.glob("*_detailed.json")) if ocr_dir.exists() else []
    text_files = list(ocr_dir.glob("*_text.txt")) if ocr_dir.exists() else []
    
    print(f"OCR Results Summary:")
    print(f"  Total files: {len(all_files)}")
    print(f"  Classification files: {len(classification_files)}")
    print(f"  Detailed JSON files: {len(detailed_files)}")
    print(f"  Text files: {len(text_files)}")
    print()
    
    # Check for old ground truth files
    old_gt_files = list(gt_dir.glob("*_gt.txt")) if gt_dir.exists() else []
    if old_gt_files:
        print(f"Found {len(old_gt_files)} old ground truth files:")
        for gt_file in old_gt_files:
            print(f"  - {gt_file.name}")
        print()
    
    # Archive old files if needed
    if old_gt_files:
        archive_dir = gt_dir / "archive_old_gt"
        archive_dir.mkdir(exist_ok=True)
        
        print("Archiving old ground truth files...")
        for gt_file in old_gt_files:
            archive_path = archive_dir / gt_file.name
            gt_file.rename(archive_path)
            print(f"  ✓ Moved {gt_file.name} to archive/")
        print()
    
    return {
        "total_files": len(all_files),
        "classification_files": len(classification_files),
        "text_files": len(text_files),
        "detailed_files": len(detailed_files)
    }

def main():
    """Main function to clean up and create ground truth templates."""
    
    print("📋 GROUND TRUTH TEMPLATE GENERATOR")
    print("=" * 60)
    print()
    
    # Clean up folders first
    stats = clean_folders()
    
    # Create ground truth templates
    if stats["classification_files"] > 0:
        ground_truth_data = create_ground_truth_templates()
        
        print()
        print("💡 USAGE INSTRUCTIONS:")
        print("=" * 60)
        print()
        print("1. Open 'ground_truth/ground_truth_template.csv' in Excel/LibreOffice")
        print("2. Review each row and correct the category and tags manually:")
        print("   - Check if category matches the actual document type")
        print("   - Verify tags are appropriate for the document content")
        print("   - Remove 'needs_review' column when done")
        print()
        print("3. Save the corrected file as 'ground_truth/ground_truth_final.csv'")
        print()
        print("4. Run F1-score evaluation:")
        print("   python3 classification_eval.py ground_truth/ground_truth_final.csv ocr_results/")
        print()
        print("Available Categories:")
        print("  - Agreement, Cleaning, Commissioning Record, Customer Complaint")
        print("  - Decommissioning Notice, Defect Notification to Customer")
        print("  - HEC Info, Inspection, Letter, Incoming Letter, Outgoing Letter")
        print("  - Replacement Notice, Slope Work, SRIC, SS Maintenance")
        print()
        print("Available Tags:")
        print("  - Agreement (Access), Agreement (Fire Services), Agreement (Ventilation)")
        print("  - Defect (Access), Defect (Civil), Defect (Fire Services), Defect (Ventilation)")
        print("  - VSC/VIC")
        
    else:
        print("❌ No classification files found!")
        print("Please run OCR processing first:")
        print("  python3 process_pdfs.py /path/to/pdfs/ en")

if __name__ == "__main__":
    main()
