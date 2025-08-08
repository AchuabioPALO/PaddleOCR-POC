#!/usr/bin/env python3
"""
Simple PaddleOCR POC - PDF Processing Script

This script processes PDF files using PaddleOCR for English and Chinese text extraction.
"""

import os
import sys
import cv2
import fitz  # pymupdf
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import json
from pathlib import Path
import time
import io

class PDFProcessor:
    def __init__(self, output_dir="ocr_results"):
        """Initialize PaddleOCR for English and Chinese text recognition."""
        print("Initializing PaddleOCR...")
        # Initialize PaddleOCR for English and Chinese
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        self.ocr_chinese = PaddleOCR(use_textline_orientation=True, lang='ch')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
    def pdf_to_images(self, pdf_path):
        """Convert PDF pages to images."""
        print(f"Converting PDF to images: {pdf_path}")
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Get page as PIL image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to numpy array for OpenCV/PaddleOCR
            img_array = np.array(img)
            images.append(img_array)
            
        doc.close()
        print(f"Converted {len(images)} pages to images")
        return images
    
    def extract_text_paddleocr(self, image, lang='en'):
        """Extract text from image using PaddleOCR."""
        ocr_engine = self.ocr if lang == 'en' else self.ocr_chinese
        result = ocr_engine.predict(image)
        
        texts = []
        if result and len(result) > 0:
            ocr_result = result[0]
            
            # Extract texts, scores, and bounding boxes
            rec_texts = ocr_result.get('rec_texts', [])
            rec_scores = ocr_result.get('rec_scores', [])
            rec_polys = ocr_result.get('rec_polys', [])
            
            for i, text in enumerate(rec_texts):
                confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                bbox = rec_polys[i].tolist() if i < len(rec_polys) else []
                
                texts.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        return texts
    
    def merge_ocr_results(self, en_results, ch_results):
        """Intelligently merge English and Chinese OCR results."""
        import re
        
        # Simple approach: prefer Chinese OCR for text containing Chinese characters,
        # English OCR for everything else
        merged_results = []
        
        # Create a set of all detected text to avoid duplicates
        all_texts = []
        
        # Process English results
        for result in en_results:
            text = result['text'].strip()
            if text and not any(self.contains_chinese(existing) and self.contains_chinese(text) for existing in all_texts):
                all_texts.append(text)
                merged_results.append({
                    'text': text,
                    'confidence': result['confidence'],
                    'source': 'en'
                })
        
        # Process Chinese results - add only if significantly different or contains Chinese
        for result in ch_results:
            text = result['text'].strip()
            if text and (self.contains_chinese(text) or not any(self.text_similarity(text, existing) > 0.8 for existing in all_texts)):
                if text not in all_texts:  # Avoid exact duplicates
                    all_texts.append(text)
                    merged_results.append({
                        'text': text,
                        'confidence': result['confidence'],
                        'source': 'ch'
                    })
        
        return merged_results
    
    def contains_chinese(self, text):
        """Check if text contains Chinese characters."""
        import re
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def text_similarity(self, text1, text2):
        """Calculate basic text similarity (simple character overlap)."""
        if not text1 or not text2:
            return 0.0
        
        text1_chars = set(text1.lower())
        text2_chars = set(text2.lower())
        
        intersection = len(text1_chars.intersection(text2_chars))
        union = len(text1_chars.union(text2_chars))
        
        return intersection / union if union > 0 else 0.0
    
    def process_pdf(self, pdf_path, lang='en'):
        """Process a single PDF file."""
        pdf_name = Path(pdf_path).stem
        print(f"\nProcessing PDF: {pdf_name}")
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        
        all_results = []
        total_text = []
        
        for page_num, image in enumerate(images):
            print(f"Processing page {page_num + 1}/{len(images)}...")
            
            if lang == 'both':
                # Start with English OCR
                en_results = self.extract_text_paddleocr(image, 'en')
                
                # Check if any English OCR results contain Chinese characters
                needs_chinese_ocr = any(self.contains_chinese(result['text']) for result in en_results)
                
                if needs_chinese_ocr:
                    # Only run Chinese OCR if we detected Chinese characters
                    print(f"  Detected Chinese characters on page {page_num + 1}, running Chinese OCR...")
                    ch_results = self.extract_text_paddleocr(image, 'ch')
                    combined_results = self.merge_ocr_results(en_results, ch_results)
                    
                    page_result = {
                        'page': page_num + 1,
                        'english_results': en_results,
                        'chinese_results': ch_results,
                        'combined_results': combined_results
                    }
                else:
                    # Use only English results if no Chinese detected
                    combined_results = en_results
                    page_result = {
                        'page': page_num + 1,
                        'english_results': en_results,
                        'combined_results': combined_results
                    }
                
                for result in combined_results:
                    total_text.append(result['text'])
                    
            else:
                # Use only the specified language
                results = self.extract_text_paddleocr(image, lang)
                
                page_result = {
                    'page': page_num + 1,
                    f'{lang}_results': results
                }
                
                # Collect text without language prefixes to avoid duplication
                for result in results:
                    total_text.append(result['text'])
            
            all_results.append(page_result)
        
        # Save detailed JSON results
        json_output_path = self.output_dir / f"{pdf_name}_detailed.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # Save simple text output
        text_output_path = self.output_dir / f"{pdf_name}_text.txt"
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(total_text))
        
        print(f"Results saved:")
        print(f"  - Detailed JSON: {json_output_path}")
        print(f"  - Text file: {text_output_path}")
        
        return all_results, total_text
    
    def process_directory(self, pdf_dir, lang='en'):
        """Process all PDF files in a directory."""
        pdf_dir = Path(pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                start_time = time.time()
                self.process_pdf(pdf_file, lang)
                end_time = time.time()
                print(f"Processing time: {end_time - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_pdfs.py <pdf_path_or_directory> [lang]")
        print("  pdf_path_or_directory: Path to PDF file or directory containing PDFs")
        print("  lang: 'en' for English (default), 'ch' for Chinese, 'both' for both models")
        sys.exit(1)
    
    input_path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else 'en'
    
    if lang not in ['en', 'ch', 'both']:
        print("Error: Language must be 'en', 'ch', or 'both'")
        sys.exit(1)
    
    if not os.path.exists(input_path):
        print(f"Error: Path {input_path} does not exist")
        sys.exit(1)
    
    processor = PDFProcessor()
    
    if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        # Process single PDF file
        processor.process_pdf(input_path, lang)
    elif os.path.isdir(input_path):
        # Process directory of PDFs
        processor.process_directory(input_path, lang)
    else:
        print(f"Error: {input_path} is not a valid PDF file or directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
