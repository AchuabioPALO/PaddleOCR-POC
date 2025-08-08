#!/usr/bin/env python3
"""
Manual Ground Truth Annotation Helper
Assists with reviewing and editing ground truth files
"""

import json
import os
from pathlib import Path

def show_document_context(doc_name):
    """Show context for a document to help with manual annotation"""
    
    print(f"📄 DOCUMENT: {doc_name}")
    print("=" * 60)
    
    # Show OCR extracted text
    text_file = Path(f"ocr_results/{doc_name}_text.txt")
    if text_file.exists():
        print("📝 EXTRACTED TEXT:")
        print("-" * 30)
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
            # Show first 500 characters
            if len(text) > 500:
                print(text[:500] + "...")
            else:
                print(text)
        print()
    
    # Show current classification
    classification_file = Path(f"ocr_results/{doc_name}_classification.json")
    if classification_file.exists():
        print("🤖 CURRENT OCR CLASSIFICATION:")
        print("-" * 30)
        with open(classification_file, 'r', encoding='utf-8') as f:
            classification = json.load(f)
        print(f"Category: {classification.get('category', 'Unknown')}")
        print(f"Tags: {classification.get('tags', [])}")
        print(f"Confidence: {classification.get('confidence', 0):.3f}")
        print(f"Keywords: {classification.get('keywords_found', [])}")
        print()
    
    # Show current ground truth
    gt_file = Path(f"ground_truth/{doc_name}_gt.json")
    if gt_file.exists():
        print("✏️  CURRENT GROUND TRUTH:")
        print("-" * 30)
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        print(f"Category: '{gt.get('category', '')}'")
        print(f"Tags: {gt.get('tags', [])}")
        print()
    
    print("📚 AVAILABLE OPTIONS:")
    print("-" * 30)
    print("Categories:")
    categories = ["Agreement", "Cleaning", "Commissioning Record", "Customer Complaint",
                 "Decommissioning Notice", "Defect Notification to Customer", "HEC Info",
                 "Inspection", "Letter", "Replacement Notice", "Slope Work", "SRIC", "SS Maintenance"]
    for i, cat in enumerate(categories, 1):
        print(f"  {i:2d}. {cat}")
    
    print("\nTags:")
    tags = ["Agreement (Access)", "Agreement (Fire)", "Agreement (Ventilation)",
           "Defect (Access)", "Defect (Civil)", "Defect (Fire)", "Defect (Ventilation)", "VSC/VIC"]
    for i, tag in enumerate(tags, 1):
        print(f"  {i}. {tag}")
    
    print("\n" + "=" * 60)
    print(f"📂 Original PDF location: Look in /home/azureuser/PDFs/ folders")
    print(f"✏️  Edit file: ground_truth/{doc_name}_gt.json")
    print("=" * 60)

def list_documents():
    """List all documents that need annotation"""
    gt_dir = Path("ground_truth")
    gt_files = list(gt_dir.glob("*_gt.json"))
    
    print("📋 DOCUMENTS REQUIRING MANUAL ANNOTATION:")
    print("=" * 50)
    
    for i, gt_file in enumerate(sorted(gt_files), 1):
        doc_name = gt_file.stem.replace("_gt", "")
        print(f"{i:2d}. {doc_name}")
    
    return [f.stem.replace("_gt", "") for f in sorted(gt_files)]

def main():
    print("🔍 MANUAL GROUND TRUTH ANNOTATION HELPER")
    print("=" * 50)
    print()
    
    documents = list_documents()
    print(f"\nTotal documents to review: {len(documents)}")
    print()
    
    while True:
        print("Options:")
        print("1. Show context for a specific document")
        print("2. List all documents")
        print("3. Exit")
        
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == "1":
            print("\nAvailable documents:")
            for i, doc in enumerate(documents, 1):
                print(f"{i:2d}. {doc}")
            
            try:
                doc_num = int(input(f"\nEnter document number (1-{len(documents)}): ")) - 1
                if 0 <= doc_num < len(documents):
                    print("\n")
                    show_document_context(documents[doc_num])
                    input("\nPress Enter to continue...")
                else:
                    print("Invalid document number!")
            except ValueError:
                print("Please enter a valid number!")
        
        elif choice == "2":
            print()
            list_documents()
            input("\nPress Enter to continue...")
        
        elif choice == "3":
            break
        
        else:
            print("Invalid choice!")
        
        print()

if __name__ == "__main__":
    main()
