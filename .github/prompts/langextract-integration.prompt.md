---
mode: agent
---

## Task: Integrate Document Classification into OCR POC

### Objective
Extend the existing PaddleOCR POC to automatically classify documents into predefined categories and assign relevant tags based on extracted text content.

### Requirements

#### 1. Document Categories (15 types)
- HEC Info, Safety Rules Notice, Decommissioning Notice, Replacement Notice
- SRIC, Letter, Incoming Letter, Outgoing Letter
- Technical Record, Commissioning Record, Maintenance Record
- Cleaning, Inspection, S/S Maintenance, Slope Work

#### 2. Document Tags (8 types)
- Agreement (Access/Fire Services/Ventilation)
- Defect (Access/Civil/Fire Services/Ventilation)
- VSC/VIC

#### 3. Technical Implementation
- Extend existing `process_pdfs.py` with classification module
- Use keyword-based classification for POC simplicity
- Support bilingual content (English/Chinese)
- Output enhanced JSON with classification results

### Implementation Plan

#### Phase 1: Core Classification Module (2-3 hours)
```python
# Create document_classifier.py
class DocumentClassifier:
    def __init__(self):
        self.category_keywords = {
            "Letter": ["dear", "yours", "sincerely", "ref:"],
            "Agreement": ["agreement", "access", "fire", "ventilation"],
            "Maintenance Record": ["maintenance", "service", "repair"],
            # ... other categories
        }
    
    def classify_document(self, text: str, filename: str) -> dict:
        # Keyword matching logic
        # Return {"category": str, "tags": list, "confidence": float}
```

#### Phase 2: Integration (1-2 hours)
- Modify `process_pdfs.py` to include classification
- Update JSON output format to include classification results
- Add classification summary to batch processing

#### Phase 3: Evaluation (1 hour)
- Create ground truth file for test documents
- Implement classification accuracy metrics
- Generate confusion matrix for categories

### Enhanced Output Format
```json
{
  "filename": "841-A-0032-Letter_Agreement.pdf",
  "ocr_text": "...",
  "classification": {
    "category": "Letter",
    "tags": ["Agreement (Access)"],
    "confidence": 0.85,
    "keywords_found": ["dear", "agreement", "access"]
  },
  "processing_time": 2.3
}
```

### Success Criteria
1. **Functionality**: Classify 90% of test documents into correct categories
2. **Performance**: Process classification in <0.5 seconds per document
3. **Integration**: Seamless addition to existing OCR pipeline
4. **Evaluation**: Generate classification accuracy report with F1 scores
5. **Usability**: CSV summary report for business users

### Constraints
- POC scope: Keyword-based classification (no ML training)
- Maintain existing OCR functionality
- Support existing PDF directory structure
- Output backward compatible with current evaluation scripts

### Deliverables
1. `document_classifier.py` - Core classification module
2. Enhanced `process_pdfs.py` with classification
3. `classification_eval.py` - Accuracy evaluation script
4. Ground truth classification file for test documents
5. Updated README with classification usage examples

### Testing Strategy
- Use existing 4 test documents for initial validation
- Create ground truth labels for Agreement documents
- Measure both OCR F1 scores and classification accuracy
- Test on bilingual content (English/Chinese keywords)

### Implementation Priority
1. **High**: Basic category classification
2. **Medium**: Tag assignment
3. **Low**: Confidence scoring and advanced features

This POC focuses on demonstrating feasibility rather than production-ready accuracy.

