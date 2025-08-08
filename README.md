# PaddleOCR POC with LangExtract ML Classification

A proof-of-concept for processing PDF documents using PaddleOCR for text extraction and LangExtract ML-based classification for automatic document categorization and tagging.

## Approach

This POC combines two complementary technologies:

1. **PaddleOCR**: Robust OCR engine for text extraction from PDF documents
2. **LangExtract**: ML-based document classifier using Hugging Face transformers (DistilBERT + Cardiff NLP) with enhanced pattern matching for accurate document categorization and tag detection

### Why LangExtract?

- **ML-Based Classification**: Uses pre-trained transformer models instead of rule-based keyword matching
- **Hybrid Approach**: Combines ML predictions with enhanced pattern matching for better accuracy
- **Confidence Scoring**: Provides realistic confidence scores for both categories and tags
- **Context-Aware**: Understands document context rather than just keyword presence

## Document Categories

- **Agreement**: Contractual documents
- **Inspection**: Technical inspection reports  
- **Cleaning**: Maintenance and cleaning records
- **Defect Notification to Customer**: Issue reporting documents
- **Incoming Letter**: Received correspondence
- **Outgoing Letter**: Sent correspondence

## Document Tags

- **Agreement (Access)**: Access-related agreements
- **Agreement (Ventilation)**: Ventilation system agreements
- **Defect (Access)**: Access-related issues
- **Defect (Civil)**: Civil engineering defects
- **VSC/VIC**: Visual control and inspection documents

## Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd paddleocr-poc

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (CPU-only PyTorch for compatibility)
pip install -r requirements.txt
```

## Usage

### 1. Process Documents with LangExtract Classification

Process a single PDF:
```bash
python process_pdfs.py path/to/document.pdf
```

Process multiple PDFs in a directory:
```bash
python process_pdfs.py path/to/pdf/directory/
```

Process with specific language mode:
```bash
python process_pdfs.py path/to/document.pdf en    # English only
python process_pdfs.py path/to/document.pdf ch    # Chinese only  
python process_pdfs.py path/to/document.pdf both  # Bilingual
```

### 2. Evaluate Classification Performance

Run F1-score evaluation against ground truth:
```bash
python simple_f1_eval.py
```

This compares OCR classification results in `ocr_results/` against ground truth files in `ground_truth/`.

### 3. Convert Ground Truth Format (if needed)

Convert text ground truth files to JSON format:
```bash
python convert_gt_format.py
```

## Output Structure

### Individual Document Results
Each processed PDF generates:
- `*_detailed.json`: Complete OCR and classification results
- `*_classification.json`: Classification summary only
- `*_text.txt`: Extracted plain text

### Classification Output Format
```json
{
  "category": "Inspection",
  "tags": [
    "Defect (Access)",
    "VSC/VIC"
  ],
  "confidence": 0.38,
  "detailed_tags": [
    {
      "tag": "Defect (Access)",
      "confidence": 0.4,
      "method": "LangExtract ML+Pattern"
    }
  ],
  "method": "LangExtract"
}
```

### Evaluation Results
F1-score evaluation generates:
- `f1_evaluation_results.json`: Detailed performance metrics
- Console output with per-category and per-tag F1-scores

## Current Performance

Based on test evaluation with 11 documents:

**Category Classification:**
- Overall Accuracy: 81.8%

**Tag Classification (F1-Scores):**
- Agreement (Access): 1.000
- Agreement (Ventilation): 1.000  
- Defect (Access): 0.933
- Defect (Civil): 0.923
- VSC/VIC: 1.000
- **Macro-averaged F1: 0.971**

## Key Files

- `process_pdfs.py`: Main processing pipeline
- `langextract_classifier.py`: LangExtract ML classifier implementation
- `simple_f1_eval.py`: F1-score evaluation script
- `convert_gt_format.py`: Ground truth format converter
- `requirements.txt`: Python dependencies

## Architecture

```
PDF Input → PaddleOCR → Text Extraction → LangExtract Classifier → JSON Output
                                              ↓
                                    ML Classification + Pattern Matching
                                              ↓
                                    Category + Tags + Confidence Scores
```

The LangExtract classifier uses:
1. **Hugging Face DistilBERT** for semantic understanding
2. **Cardiff NLP models** for document classification
3. **Enhanced regex patterns** for tag detection
4. **Confidence scoring** based on ML predictions and pattern matches

## Dependencies

Key Python packages:
- `paddlepaddle` - Deep learning framework for OCR
- `paddleocr` - OCR text extraction engine
- `torch` - PyTorch (CPU-only version for compatibility)
- `transformers` - Hugging Face transformers for ML classification
- `scikit-learn` - Performance evaluation metrics
- `opencv-python` - Image processing
- `pymupdf` - PDF handling

## Notes

- Uses CPU-only PyTorch to avoid CUDA dependency issues
- Processing time: ~2-3 seconds per page including ML classification
- Supports bilingual text extraction (English + Chinese)
- Confidence thresholds: Category (0.3), Tags (varies by pattern strength)
  - Text file: ocr_results/841-A-0032-Letter 19761029_Agreement_text.txt
Processing time: 15.32 seconds
```

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce image resolution in `pdf_to_images()` method
2. **Slow processing**: Consider enabling GPU support or reducing image quality
3. **Import errors**: Ensure all dependencies are installed in the virtual environment

### Getting Help
Check the console output for detailed error messages and processing status.
