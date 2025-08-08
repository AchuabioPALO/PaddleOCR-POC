#!/usr/bin/env python3
from document_classifier import DocumentClassifier
import json

# Load the text from Document 1
with open('ocr_results/841-A-0010_Letter.Page 1-1_text.txt', 'r') as f:
    text = f.read()

classifier = DocumentClassifier()
result = classifier.classify_document(text, '841-A-0010_Letter.Page 1-1.pdf')

print('🔄 Updated Classification for Document 1:')
print(f'Category: {result["category"]} (confidence: {result["confidence"]:.3f})')
print(f'Tags: {result["tags"]}')
print(f'Keywords found: {result["keywords_found"][:10]}')
print(f'Expected: Should get Agreement (Access) tag for electrical supply')
