#!/usr/bin/env python3
"""
Classification Evaluation Script for PaddleOCR POC

This script evaluates the accuracy of document classification by comparing
predicted classifications against ground truth labels.
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, List
import argparse
from collections import defaultdict, Counter


class ClassificationEvaluator:
    def __init__(self):
        """Initialize the classification evaluator."""
        self.ground_truth = {}
        self.predictions = {}
        
    def load_ground_truth(self, ground_truth_file: str):
        """Load ground truth classifications from file."""
        ground_truth_path = Path(ground_truth_file)
        
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
        
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            if ground_truth_file.endswith('.csv'):
                self.ground_truth = self._load_csv_ground_truth(f)
            else:
                self.ground_truth = json.load(f)
        
        print(f"Loaded ground truth for {len(self.ground_truth)} documents")
        
    def _load_csv_ground_truth(self, file_handle):
        """Load ground truth from CSV format."""
        reader = csv.DictReader(file_handle)
        ground_truth = {}
        
        for row in reader:
            filename = row['filename']
            category = row['category']
            tags = [tag.strip() for tag in row.get('tags', '').split(';') if tag.strip()]
            
            ground_truth[filename] = {
                'category': category,
                'tags': tags
            }
        
        return ground_truth
        
    def load_predictions(self, results_dir: str):
        """Load classification predictions from OCR results directory."""
        results_path = Path(results_dir)
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        # Look for classification files
        classification_files = list(results_path.glob("*_classification.json"))
        detailed_files = list(results_path.glob("*_detailed.json"))
        
        predictions = {}
        
        # Load from individual classification files
        for class_file in classification_files:
            with open(class_file, 'r', encoding='utf-8') as f:
                classification = json.load(f)
                filename = class_file.name.replace('_classification.json', '.pdf')
                predictions[filename] = classification
        
        # Load from detailed files if no separate classification files
        if not classification_files and detailed_files:
            for detail_file in detailed_files:
                with open(detail_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'classification' in data and data['classification']:
                        filename = data.get('filename', detail_file.name.replace('_detailed.json', '.pdf'))
                        predictions[filename] = data['classification']
        
        self.predictions = predictions
        print(f"Loaded predictions for {len(self.predictions)} documents")
        
    def calculate_metrics(self):
        """Calculate classification accuracy metrics."""
        if not self.ground_truth or not self.predictions:
            raise ValueError("Both ground truth and predictions must be loaded")
        
        # Find common documents
        common_docs = set(self.ground_truth.keys()) & set(self.predictions.keys())
        
        if not common_docs:
            raise ValueError("No common documents found between ground truth and predictions")
        
        print(f"Evaluating {len(common_docs)} common documents")
        
        # Category classification metrics
        category_metrics = self._calculate_category_metrics(common_docs)
        
        # Tag classification metrics  
        tag_metrics = self._calculate_tag_metrics(common_docs)
        
        # Overall metrics
        overall_metrics = self._calculate_overall_metrics(common_docs)
        
        return {
            'category_metrics': category_metrics,
            'tag_metrics': tag_metrics,
            'overall_metrics': overall_metrics,
            'document_count': len(common_docs),
            'common_documents': list(common_docs)
        }
    
    def _calculate_category_metrics(self, common_docs):
        """Calculate category classification metrics."""
        y_true = []
        y_pred = []
        correct_predictions = 0
        
        category_confusion = defaultdict(lambda: defaultdict(int))
        
        for doc in common_docs:
            true_category = self.ground_truth[doc]['category']
            pred_category = self.predictions[doc].get('category', 'Unknown')
            
            y_true.append(true_category)
            y_pred.append(pred_category)
            
            category_confusion[true_category][pred_category] += 1
            
            if true_category == pred_category:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(common_docs)
        
        # Calculate per-category metrics
        categories = set(y_true) | set(y_pred)
        category_stats = {}
        
        for category in categories:
            tp = category_confusion[category][category]
            fp = sum(category_confusion[other][category] for other in categories if other != category)
            fn = sum(category_confusion[category][other] for other in categories if other != category)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            category_stats[category] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': tp + fn
            }
        
        return {
            'accuracy': accuracy,
            'per_category_metrics': category_stats,
            'confusion_matrix': dict(category_confusion)
        }
    
    def _calculate_tag_metrics(self, common_docs):
        """Calculate tag classification metrics."""
        all_true_tags = set()
        all_pred_tags = set()
        
        tag_tp = defaultdict(int)
        tag_fp = defaultdict(int)
        tag_fn = defaultdict(int)
        
        for doc in common_docs:
            true_tags = set(self.ground_truth[doc].get('tags', []))
            pred_tags = set(self.predictions[doc].get('tags', []))
            
            all_true_tags.update(true_tags)
            all_pred_tags.update(pred_tags)
            
            # Calculate TP, FP, FN for each tag
            for tag in true_tags | pred_tags:
                if tag in true_tags and tag in pred_tags:
                    tag_tp[tag] += 1
                elif tag in pred_tags and tag not in true_tags:
                    tag_fp[tag] += 1
                elif tag in true_tags and tag not in pred_tags:
                    tag_fn[tag] += 1
        
        # Calculate metrics for each tag
        tag_stats = {}
        all_tags = all_true_tags | all_pred_tags
        
        for tag in all_tags:
            tp = tag_tp[tag]
            fp = tag_fp[tag]
            fn = tag_fn[tag]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            tag_stats[tag] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': tp + fn
            }
        
        # Micro and macro averages
        total_tp = sum(tag_tp.values())
        total_fp = sum(tag_fp.values())
        total_fn = sum(tag_fn.values())
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        macro_precision = sum(stats['precision'] for stats in tag_stats.values()) / len(tag_stats) if tag_stats else 0.0
        macro_recall = sum(stats['recall'] for stats in tag_stats.values()) / len(tag_stats) if tag_stats else 0.0
        macro_f1 = sum(stats['f1_score'] for stats in tag_stats.values()) / len(tag_stats) if tag_stats else 0.0
        
        return {
            'per_tag_metrics': tag_stats,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
    
    def _calculate_overall_metrics(self, common_docs):
        """Calculate overall classification metrics."""
        confidence_scores = []
        
        for doc in common_docs:
            confidence = self.predictions[doc].get('confidence', 0.0)
            confidence_scores.append(confidence)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        confidence_ranges = {
            'high (>=0.7)': sum(1 for c in confidence_scores if c >= 0.7),
            'medium (0.3-0.7)': sum(1 for c in confidence_scores if 0.3 <= c < 0.7),
            'low (<0.3)': sum(1 for c in confidence_scores if c < 0.3)
        }
        
        return {
            'average_confidence': avg_confidence,
            'confidence_distribution': confidence_ranges
        }
    
    def generate_report(self, metrics: Dict, output_file: str = None):
        """Generate a human-readable evaluation report."""
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("DOCUMENT CLASSIFICATION EVALUATION REPORT")
        report_lines.append("=" * 60)
        
        # Overall summary
        report_lines.append(f"\nDocument Count: {metrics['document_count']}")
        report_lines.append(f"Average Confidence: {metrics['overall_metrics']['average_confidence']:.3f}")
        
        # Category metrics
        report_lines.append("\n" + "=" * 40)
        report_lines.append("CATEGORY CLASSIFICATION METRICS")
        report_lines.append("=" * 40)
        
        cat_metrics = metrics['category_metrics']
        report_lines.append(f"Overall Accuracy: {cat_metrics['accuracy']:.3f}")
        
        report_lines.append("\nPer-Category Performance:")
        for category, stats in cat_metrics['per_category_metrics'].items():
            report_lines.append(f"  {category}:")
            report_lines.append(f"    Precision: {stats['precision']:.3f}")
            report_lines.append(f"    Recall: {stats['recall']:.3f}")
            report_lines.append(f"    F1-Score: {stats['f1_score']:.3f}")
            report_lines.append(f"    Support: {stats['support']}")
        
        # Tag metrics
        report_lines.append("\n" + "=" * 40)
        report_lines.append("TAG CLASSIFICATION METRICS")
        report_lines.append("=" * 40)
        
        tag_metrics = metrics['tag_metrics']
        report_lines.append(f"Micro-averaged F1: {tag_metrics['micro_f1']:.3f}")
        report_lines.append(f"Macro-averaged F1: {tag_metrics['macro_f1']:.3f}")
        
        if tag_metrics['per_tag_metrics']:
            report_lines.append("\nPer-Tag Performance:")
            for tag, stats in tag_metrics['per_tag_metrics'].items():
                if stats['support'] > 0:  # Only show tags that appear in ground truth
                    report_lines.append(f"  {tag}:")
                    report_lines.append(f"    Precision: {stats['precision']:.3f}")
                    report_lines.append(f"    Recall: {stats['recall']:.3f}")
                    report_lines.append(f"    F1-Score: {stats['f1_score']:.3f}")
                    report_lines.append(f"    Support: {stats['support']}")
        
        # Confidence distribution
        report_lines.append("\n" + "=" * 40)
        report_lines.append("CONFIDENCE DISTRIBUTION")
        report_lines.append("=" * 40)
        
        conf_dist = metrics['overall_metrics']['confidence_distribution']
        for range_name, count in conf_dist.items():
            percentage = (count / metrics['document_count']) * 100
            report_lines.append(f"  {range_name}: {count} documents ({percentage:.1f}%)")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Evaluation report saved to: {output_file}")
        else:
            print(report_content)
        
        return report_content


def main():
    parser = argparse.ArgumentParser(description='Evaluate document classification accuracy')
    parser.add_argument('ground_truth', help='Path to ground truth file (JSON or CSV)')
    parser.add_argument('results_dir', help='Path to OCR results directory')
    parser.add_argument('--output', '-o', help='Output file for evaluation report')
    parser.add_argument('--json-output', help='Save detailed metrics as JSON')
    
    args = parser.parse_args()
    
    try:
        evaluator = ClassificationEvaluator()
        
        # Load data
        evaluator.load_ground_truth(args.ground_truth)
        evaluator.load_predictions(args.results_dir)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics()
        
        # Generate report
        report_output = args.output or "classification_evaluation_report.txt"
        evaluator.generate_report(metrics, report_output)
        
        # Save detailed metrics as JSON if requested
        if args.json_output:
            with open(args.json_output, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"Detailed metrics saved to: {args.json_output}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"Category Accuracy: {metrics['category_metrics']['accuracy']:.3f}")
        print(f"Tag F1 (Micro): {metrics['tag_metrics']['micro_f1']:.3f}")
        print(f"Tag F1 (Macro): {metrics['tag_metrics']['macro_f1']:.3f}")
        print(f"Average Confidence: {metrics['overall_metrics']['average_confidence']:.3f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
