#!/usr/bin/env python3
"""
F1 Score Evaluation Script for OCR Results

This script calculates F1 scores to evaluate OCR performance by comparing
OCR output with ground truth text.
"""

import os
import sys
import json
import re
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class OCRValidator:
    def __init__(self):
        """Initialize the OCR validator."""
        self.results = []
        
    def clean_text(self, text):
        """Clean and normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into words."""
        clean_text = self.clean_text(text)
        words = clean_text.split()
        return set(words)  # Use set for word-level comparison
    
    def calculate_word_level_f1(self, predicted_text, ground_truth_text):
        """Calculate word-level F1 score."""
        pred_words = self.tokenize_text(predicted_text)
        gt_words = self.tokenize_text(ground_truth_text)
        
        if not gt_words:
            return 0.0, 0.0, 0.0  # precision, recall, f1
        
        # Calculate metrics
        true_positives = len(pred_words.intersection(gt_words))
        false_positives = len(pred_words - gt_words)
        false_negatives = len(gt_words - pred_words)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def calculate_character_level_f1(self, predicted_text, ground_truth_text):
        """Calculate character-level F1 score."""
        pred_chars = set(self.clean_text(predicted_text).replace(' ', ''))
        gt_chars = set(self.clean_text(ground_truth_text).replace(' ', ''))
        
        if not gt_chars:
            return 0.0, 0.0, 0.0
        
        true_positives = len(pred_chars.intersection(gt_chars))
        false_positives = len(pred_chars - gt_chars)
        false_negatives = len(gt_chars - pred_chars)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def load_ground_truth(self, ground_truth_path):
        """Load ground truth text from file."""
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_ocr_results(self, ocr_results_path):
        """Load OCR results from text file."""
        with open(ocr_results_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def evaluate_file(self, ocr_results_path, ground_truth_path):
        """Evaluate a single file."""
        print(f"Evaluating: {Path(ocr_results_path).name}")
        
        try:
            # Load texts
            ocr_text = self.load_ocr_results(ocr_results_path)
            gt_text = self.load_ground_truth(ground_truth_path)
            
            # Calculate word-level metrics
            word_precision, word_recall, word_f1 = self.calculate_word_level_f1(ocr_text, gt_text)
            
            # Calculate character-level metrics
            char_precision, char_recall, char_f1 = self.calculate_character_level_f1(ocr_text, gt_text)
            
            result = {
                'file': Path(ocr_results_path).name,
                'word_level': {
                    'precision': word_precision,
                    'recall': word_recall,
                    'f1': word_f1
                },
                'character_level': {
                    'precision': char_precision,
                    'recall': char_recall,
                    'f1': char_f1
                }
            }
            
            self.results.append(result)
            
            print(f"  Word-level F1: {word_f1:.3f} (P: {word_precision:.3f}, R: {word_recall:.3f})")
            print(f"  Char-level F1: {char_f1:.3f} (P: {char_precision:.3f}, R: {char_recall:.3f})")
            
            return result
            
        except Exception as e:
            print(f"Error evaluating {ocr_results_path}: {str(e)}")
            return None
    
    def evaluate_directory(self, ocr_results_dir, ground_truth_dir):
        """Evaluate all files in directories."""
        ocr_dir = Path(ocr_results_dir)
        gt_dir = Path(ground_truth_dir)
        
        # Find all OCR result text files
        ocr_files = list(ocr_dir.glob("*_text.txt"))
        
        if not ocr_files:
            print(f"No OCR result files (*_text.txt) found in {ocr_dir}")
            return
        
        print(f"Found {len(ocr_files)} OCR result files to evaluate")
        
        for ocr_file in ocr_files:
            # Try to find corresponding ground truth file
            base_name = ocr_file.stem.replace('_text', '')
            
            # Look for ground truth files with various naming patterns
            gt_candidates = [
                gt_dir / f"{base_name}_gt.txt",
                gt_dir / f"{base_name}_groundtruth.txt",
                gt_dir / f"{base_name}.txt",
                gt_dir / f"gt_{base_name}.txt"
            ]
            
            gt_file = None
            for candidate in gt_candidates:
                if candidate.exists():
                    gt_file = candidate
                    break
            
            if gt_file:
                self.evaluate_file(ocr_file, gt_file)
            else:
                print(f"Warning: No ground truth file found for {ocr_file.name}")
    
    def generate_report(self, output_path="evaluation_report.json"):
        """Generate evaluation report."""
        if not self.results:
            print("No results to report")
            return
        
        # Calculate overall statistics
        word_f1_scores = [r['word_level']['f1'] for r in self.results]
        char_f1_scores = [r['character_level']['f1'] for r in self.results]
        
        overall_stats = {
            'total_files': len(self.results),
            'word_level_stats': {
                'mean_f1': np.mean(word_f1_scores),
                'std_f1': np.std(word_f1_scores),
                'min_f1': np.min(word_f1_scores),
                'max_f1': np.max(word_f1_scores)
            },
            'character_level_stats': {
                'mean_f1': np.mean(char_f1_scores),
                'std_f1': np.std(char_f1_scores),
                'min_f1': np.min(char_f1_scores),
                'max_f1': np.max(char_f1_scores)
            }
        }
        
        report = {
            'overall_statistics': overall_stats,
            'individual_results': self.results
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"\n=== EVALUATION REPORT ===")
        print(f"Total files evaluated: {overall_stats['total_files']}")
        print(f"\nWord-level Performance:")
        print(f"  Mean F1: {overall_stats['word_level_stats']['mean_f1']:.3f} ± {overall_stats['word_level_stats']['std_f1']:.3f}")
        print(f"  Range: {overall_stats['word_level_stats']['min_f1']:.3f} - {overall_stats['word_level_stats']['max_f1']:.3f}")
        print(f"\nCharacter-level Performance:")
        print(f"  Mean F1: {overall_stats['character_level_stats']['mean_f1']:.3f} ± {overall_stats['character_level_stats']['std_f1']:.3f}")
        print(f"  Range: {overall_stats['character_level_stats']['min_f1']:.3f} - {overall_stats['character_level_stats']['max_f1']:.3f}")
        print(f"\nDetailed report saved to: {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_f1.py <ocr_results_path> <ground_truth_path> [output_report]")
        print("  ocr_results_path: Path to OCR results file or directory")
        print("  ground_truth_path: Path to ground truth file or directory")
        print("  output_report: Output report file (optional, default: evaluation_report.json)")
        sys.exit(1)
    
    ocr_path = sys.argv[1]
    gt_path = sys.argv[2]
    output_report = sys.argv[3] if len(sys.argv) > 3 else "evaluation_report.json"
    
    if not os.path.exists(ocr_path):
        print(f"Error: OCR results path {ocr_path} does not exist")
        sys.exit(1)
    
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth path {gt_path} does not exist")
        sys.exit(1)
    
    validator = OCRValidator()
    
    if os.path.isfile(ocr_path) and os.path.isfile(gt_path):
        # Evaluate single file
        validator.evaluate_file(ocr_path, gt_path)
    elif os.path.isdir(ocr_path) and os.path.isdir(gt_path):
        # Evaluate directory
        validator.evaluate_directory(ocr_path, gt_path)
    else:
        print("Error: Both paths must be files or both must be directories")
        sys.exit(1)
    
    # Generate report
    validator.generate_report(output_report)

if __name__ == "__main__":
    main()
