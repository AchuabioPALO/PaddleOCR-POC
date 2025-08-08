#!/usr/bin/env python3
"""
PaddleOCR POC Demo Script

This script demonstrates the complete pipeline:
1. Process PDFs with PaddleOCR
2. Evaluate results (if ground truth is available)
3. Generate summary report
"""

import os
import sys
import time
from pathlib import Path
import subprocess

def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Exit code: {result.returncode}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    # Configuration
    pdf_dir = "/home/azureuser/PDFs/Agreement/Un-Split"
    output_dir = "ocr_results"
    
    print("PaddleOCR POC Demo")
    print("=" * 50)
    print(f"PDF Directory: {pdf_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Check if PDF directory exists
    if not os.path.exists(pdf_dir):
        print(f"Error: PDF directory {pdf_dir} does not exist")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Process PDFs with PaddleOCR
    print(f"\nStep 1: Processing PDFs with PaddleOCR")
    success = run_command(
        f"cd /home/azureuser/paddleocr-poc && . venv/bin/activate && python process_pdfs.py {pdf_dir}",
        "PDF Processing with PaddleOCR"
    )
    
    if not success:
        print("PDF processing failed!")
        return False
    
    # Step 2: List generated files
    print(f"\nStep 2: Listing generated OCR results")
    ocr_results_dir = Path("ocr_results")
    if ocr_results_dir.exists():
        text_files = list(ocr_results_dir.glob("*_text.txt"))
        json_files = list(ocr_results_dir.glob("*_detailed.json"))
        
        print(f"Generated {len(text_files)} text files:")
        for f in text_files:
            print(f"  - {f.name}")
        
        print(f"Generated {len(json_files)} detailed JSON files:")
        for f in json_files:
            print(f"  - {f.name}")
    else:
        print("No OCR results directory found!")
        return False
    
    # Step 3: Show sample results
    print(f"\nStep 3: Showing sample OCR results")
    if text_files:
        sample_file = text_files[0]
        print(f"Sample from {sample_file.name}:")
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Show first 500 characters
                print(content[:500] + ("..." if len(content) > 500 else ""))
        except Exception as e:
            print(f"Error reading sample file: {e}")
    
    # Step 4: Generate summary statistics
    print(f"\nStep 4: Generating summary statistics")
    total_chars = 0
    total_lines = 0
    
    for text_file in text_files:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
                total_chars += len(content)
                total_lines += len(content.split('\n'))
        except Exception as e:
            print(f"Error reading {text_file}: {e}")
    
    print(f"Summary Statistics:")
    print(f"  - Total files processed: {len(text_files)}")
    print(f"  - Total characters extracted: {total_chars:,}")
    print(f"  - Total lines extracted: {total_lines:,}")
    print(f"  - Average characters per file: {total_chars / len(text_files):.0f}")
    print(f"  - Average lines per file: {total_lines / len(text_files):.0f}")
    
    print(f"\n{'='*50}")
    print("PaddleOCR POC Demo completed successfully!")
    print(f"Results are available in: {Path(output_dir).absolute()}")
    print(f"{'='*50}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
