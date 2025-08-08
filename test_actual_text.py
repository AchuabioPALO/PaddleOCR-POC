#!/usr/bin/env python3
from document_classifier import DocumentClassifier

# Load the actual extracted text
with open('ocr_results/841-A-0049_Agreement_text.txt', 'r') as f:
    actual_text = f.read()

classifier = DocumentClassifier()
result = classifier.classify_document(actual_text, '841-A-0049_Agreement.pdf')

print('Re-classification with actual OCR text:')
print(f'Category: {result["category"]} (confidence: {result["confidence"]:.3f})')
print(f'Tags: {result["tags"]}')
print(f'Top keywords: {result["keywords_found"][:8]}')
print(f'Expected: Incoming Letter | Agreement (Access)')
print(f'Match: {"✓" if result["category"] == "Incoming Letter" else "✗"}')
