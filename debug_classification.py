#!/usr/bin/env python3
from document_classifier import DocumentClassifier

# Load the actual extracted text
with open('ocr_results/841-A-0130_Agreement.Page 1-2_text.txt', 'r') as f:
    text = f.read()

print('Sample of text:')
print(text[:300])
print()

classifier = DocumentClassifier()
result = classifier.classify_document(text, '841-A-0130_Agreement.Page 1-2.pdf')

print('Classification Result:')
print(f'Category: {result["category"]} (confidence: {result["confidence"]:.3f})')
print(f'Tags: {result["tags"]}')
print(f'Keywords found: {result["keywords_found"]}')

# Check what should make it an outgoing letter
text_lower = text.lower()
outgoing_indicators = ["our ref", "msd/", "despatched", "dear sir"]
print(f'\nOutgoing letter indicators found:')
for indicator in outgoing_indicators:
    if indicator in text_lower:
        print(f'✓ Found: "{indicator}"')
    else:
        print(f'✗ Missing: "{indicator}"')
