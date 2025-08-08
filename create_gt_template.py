#!/usr/bin/env python3
"""
Ground Truth Template Creator

This script helps create ground truth templates from OCR results,
making it easier to manually correct the text for F1 evaluation.
"""

import os
import sys
from pathlib import Path
import re

def clean_ocr_text(text):
    """Clean OCR text to create a better template for ground truth editing."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove [EN] and [CH] prefixes
        line = re.sub(r'^\[EN\]\s*', '', line)
        line = re.sub(r'^\[CH\]\s*', '', line)
        
        # Skip empty lines from cleaning
        if line.strip():
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def remove_duplicates(text):
    """Remove duplicate lines that appear consecutively."""
    lines = text.split('\n')
    unique_lines = []
    prev_line = None
    
    for line in lines:
        # Skip if this line is identical to the previous one
        if line.strip() != prev_line:
            unique_lines.append(line)
        prev_line = line.strip()
    
    return '\n'.join(unique_lines)

def create_gt_template(ocr_file_path, output_dir="ground_truth"):
    """Create a ground truth template from an OCR result file."""
    ocr_path = Path(ocr_file_path)
    
    if not ocr_path.exists():
        print(f"Error: OCR file {ocr_path} does not exist")
        return False
    
    # Read OCR results
    with open(ocr_path, 'r', encoding='utf-8') as f:
        ocr_text = f.read()
    
    # Clean the text
    cleaned_text = clean_ocr_text(ocr_text)
    cleaned_text = remove_duplicates(cleaned_text)
    
    # Create ground truth filename
    base_name = ocr_path.stem.replace('_text', '')
    gt_filename = f"{base_name}_gt.txt"
    gt_path = Path(output_dir) / gt_filename
    
    # Create template with instructions
    template = f"""# Ground Truth for {base_name}
# Instructions:
# 1. Review and correct the OCR text below
# 2. Remove obvious OCR errors (garbled text from diagrams, etc.)
# 3. Fix spelling and formatting errors
# 4. Keep the main document content intact
# 5. Remove these instruction lines when done
# 6. Save the file as-is for F1 evaluation
#
# Original OCR file: {ocr_path.name}
# 
# ===== GROUND TRUTH TEXT STARTS BELOW =====

{cleaned_text}
"""
    
    # Save template
    Path(output_dir).mkdir(exist_ok=True)
    with open(gt_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"Created ground truth template: {gt_path}")
    print(f"  Original OCR file: {ocr_path}")
    print(f"  Lines in original: {len(ocr_text.split(chr(10)))}")
    print(f"  Lines in template: {len(cleaned_text.split(chr(10)))}")
    
    return True

def create_all_templates(ocr_results_dir="ocr_results", output_dir="ground_truth"):
    """Create ground truth templates for all OCR result files."""
    ocr_dir = Path(ocr_results_dir)
    
    # Find all text files
    text_files = list(ocr_dir.glob("*_text.txt"))
    
    if not text_files:
        print(f"No OCR text files found in {ocr_dir}")
        return
    
    print(f"Found {len(text_files)} OCR text files")
    print("Creating ground truth templates...\n")
    
    success_count = 0
    for text_file in text_files:
        if create_gt_template(text_file, output_dir):
            success_count += 1
    
    print(f"\nCreated {success_count}/{len(text_files)} ground truth templates")
    print(f"Templates saved in: {Path(output_dir).absolute()}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python create_gt_template.py <ocr_file>           # Create template for single file")
        print("  python create_gt_template.py --all               # Create templates for all OCR files")
        print("  python create_gt_template.py --help              # Show this help")
        sys.exit(1)
    
    if sys.argv[1] == "--help":
        print(__doc__)
        return
    
    if sys.argv[1] == "--all":
        create_all_templates()
    else:
        ocr_file = sys.argv[1]
        create_gt_template(ocr_file)

if __name__ == "__main__":
    main()
