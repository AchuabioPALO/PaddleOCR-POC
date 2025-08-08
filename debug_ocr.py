#!/usr/bin/env python3
"""
Debug script to understand PaddleOCR output format
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

def test_paddleocr():
    """Test PaddleOCR on a simple example to understand output format."""
    
    # Initialize PaddleOCR
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    # Load a PDF page as image
    pdf_path = "/home/azureuser/PDFs/Agreement/Un-Split/841-A-0032-Letter 19761029_Agreement.pdf"
    print(f"Loading PDF: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # First page only
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = pix.tobytes("ppm")
    img = Image.open(io.BytesIO(img_data))
    img_array = np.array(img)
    doc.close()
    
    print(f"Image shape: {img_array.shape}")
    
    # Run OCR
    print("Running OCR...")
    result = ocr.predict(img_array)
    
    print(f"OCR result type: {type(result)}")
    print(f"OCR result length: {len(result) if result else 0}")
    
    if result:
        ocr_result = result[0]
        print(f"First element type: {type(ocr_result)}")
        print(f"OCR result attributes: {dir(ocr_result)}")
        
        # Try to access common attributes
        if hasattr(ocr_result, 'text'):
            print(f"Text: {ocr_result.text}")
        if hasattr(ocr_result, 'boxes'):
            print(f"Boxes: {ocr_result.boxes}")
        if hasattr(ocr_result, 'rec_texts'):
            print(f"Recognition texts: {ocr_result.rec_texts}")
        if hasattr(ocr_result, 'rec_scores'):
            print(f"Recognition scores: {ocr_result.rec_scores}")
        
        # Print first few lines if available
        print(f"Full result: {ocr_result}")

if __name__ == "__main__":
    test_paddleocr()
