#!/usr/bin/env python3
"""
Test the classifier on the specific document that was misclassified
"""

from document_classifier import DocumentClassifier

# Sample text based on the document pattern you described
sample_text = """
Architectural Office
Public Works Department
The Government of Hong Kong

To: HK Electric

Re: Laying of LV Cable for Street Lighting

Dear Sir/Madam,

We confirm that the repair work has been completed and we agree with the access arrangements for the construction work. The laying of the LV cable for street lighting has been approved.

The contractor may proceed with the work as discussed. Please ensure all safety protocols are followed during the access to the site.

Yours faithfully,
Senior Engineer
Architectural Office
"""

def test_specific_document():
    classifier = DocumentClassifier()
    
    print("Testing classification on the problematic document...")
    print("=" * 60)
    
    result = classifier.classify_document(sample_text, "841-A-0049_Agreement.pdf")
    
    print(f"Document: 841-A-0049_Agreement.pdf")
    print(f"Category: {result['category']} (confidence: {result['confidence']:.3f})")
    print(f"Tags: {result['tags']}")
    print(f"Keywords found: {result['keywords_found']}")
    
    print("\nExpected vs Actual:")
    print(f"Expected Category: Incoming Letter")
    print(f"Actual Category: {result['category']}")
    print(f"Match: {'✓' if result['category'] == 'Incoming Letter' else '✗'}")
    
    print(f"\nExpected Tags: Agreement (Access)")
    print(f"Actual Tags: {result['tags']}")
    tag_match = 'Agreement (Access)' in result['tags']
    print(f"Tag Match: {'✓' if tag_match else '✗'}")

if __name__ == "__main__":
    test_specific_document()
