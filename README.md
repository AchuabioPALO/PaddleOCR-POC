# PaddleOCR POC

A simple proof-of-concept using PaddleOCR to process PDF files with English and Chinese text, and generate F1 scores to evaluate OCR performance.

## Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone/download this project
2. Install dependencies:
```bash
cd paddleocr-poc
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Quick Demo
Run the complete pipeline on sample PDFs:
```bash
python demo.py
```

### 2. Process Individual PDFs
Process a single PDF file:
```bash
python process_pdfs.py path/to/document.pdf
```

Process all PDFs in a directory:
```bash
python process_pdfs.py path/to/pdf/directory/
```

For Chinese text processing:
```bash
python process_pdfs.py path/to/document.pdf ch
```

### 3. Evaluate OCR Performance
If you have ground truth text files, you can evaluate OCR performance:
```bash
python evaluate_f1.py ocr_results/ ground_truth/ evaluation_report.json
```

## Output Files

### OCR Results
- `{filename}_text.txt` - Extracted text in simple format
- `{filename}_detailed.json` - Detailed results with confidence scores and bounding boxes

### Evaluation Results
- `evaluation_report.json` - Detailed F1 score analysis
- Console output with summary statistics

## Features

### PDF Processing
- Converts PDF pages to high-quality images
- Processes with both English and Chinese OCR models
- Extracts text with confidence scores and bounding boxes
- Saves results in multiple formats

### Evaluation
- Word-level F1 score calculation
- Character-level F1 score calculation
- Precision and recall metrics
- Statistical summary across multiple files

## Project Structure

```
paddleocr-poc/
├── process_pdfs.py       # Main PDF processing script
├── evaluate_f1.py        # F1 score evaluation script
├── demo.py              # Complete pipeline demonstration
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── ocr_results/        # Output directory (created automatically)
```

## Data Source

The primary test data is located in:
- `/home/azureuser/PDFs/Agreement/Un-Split/` - Unsplit PDF files for processing
- `/home/azureuser/PDFs/Agreement/[other folders]/` - Split versions for comparison

## Dependencies

- `paddlepaddle` - PaddlePaddle deep learning framework
- `paddleocr` - PaddleOCR library for text recognition
- `opencv-python` - Image processing
- `pillow` - Image handling
- `pymupdf` - PDF processing
- `scikit-learn` - F1 score calculation

## Performance Notes

- Processing time varies based on document size and complexity
- CPU-only processing (GPU can be enabled by modifying `use_gpu=True`)
- English and Chinese models are loaded separately for better accuracy

## Example Output

```
Processing PDF: 841-A-0032-Letter 19761029_Agreement
Converting PDF to images: /path/to/file.pdf
Converted 2 pages to images
Processing page 1/2...
Processing page 2/2...
Results saved:
  - Detailed JSON: ocr_results/841-A-0032-Letter 19761029_Agreement_detailed.json
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
