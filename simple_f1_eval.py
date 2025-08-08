#!/usr/bin/env python3
"""
Simple F1-Score Evaluation for Document Classification
Compares OCR classification results with ground truth
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, classification_report

def load_classification_data():
    """Load OCR results and ground truth data"""
    
    # Load OCR results
    ocr_results = {}
    ocr_dir = Path("ocr_results")
    for file in ocr_dir.glob("*_classification.json"):
        doc_name = file.stem.replace("_classification", "")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                ocr_results[doc_name] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ JSON error in OCR file {file}: {e}")
            continue
    
    # Load ground truth
    ground_truth = {}
    gt_dir = Path("ground_truth")
    for file in gt_dir.glob("*_gt.json"):
        doc_name = file.stem.replace("_gt", "")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                ground_truth[doc_name] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ JSON error in ground truth file {file}: {e}")
            continue
    
    return ocr_results, ground_truth

def evaluate_categories(ocr_results, ground_truth):
    """Evaluate category classification performance"""
    
    print("📊 CATEGORY CLASSIFICATION EVALUATION")
    print("=" * 50)
    
    # Collect predictions and true labels
    y_true = []
    y_pred = []
    
    for doc_name in sorted(ocr_results.keys()):
        if doc_name in ground_truth:
            true_cat = ground_truth[doc_name].get("category", "Unknown")
            pred_cat = ocr_results[doc_name].get("category", "Unknown")
            
            y_true.append(true_cat)
            y_pred.append(pred_cat)
            
            print(f"📄 {doc_name}")
            print(f"   True: {true_cat}")
            print(f"   Pred: {pred_cat}")
            print(f"   ✅ Match: {true_cat == pred_cat}")
            print()
    
    # Calculate metrics
    if y_true and y_pred:
        # Overall accuracy
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        print(f"🎯 CATEGORY ACCURACY: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print()
        
        # Detailed classification report
        print("📈 DETAILED CATEGORY METRICS:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        return accuracy, y_true, y_pred
    else:
        print("❌ No matching documents found for evaluation")
        return 0.0, [], []

def evaluate_tags(ocr_results, ground_truth):
    """Evaluate tag classification performance (multi-label)"""
    
    print("🏷️  TAG CLASSIFICATION EVALUATION")
    print("=" * 50)
    
    # Collect all unique tags
    all_tags = set()
    for doc_data in list(ocr_results.values()) + list(ground_truth.values()):
        tags = doc_data.get("tags", [])
        all_tags.update(tags)
    
    all_tags = sorted(list(all_tags))
    print(f"📋 Found {len(all_tags)} unique tags: {all_tags}")
    print()
    
    # Evaluate each tag individually (binary classification)
    tag_results = {}
    
    for tag in all_tags:
        y_true = []
        y_pred = []
        
        print(f"🔍 DEBUG: Evaluating tag '{tag}'")
        
        for doc_name in sorted(ocr_results.keys()):
            if doc_name in ground_truth:
                true_tags = ground_truth[doc_name].get("tags", [])
                pred_tags = ocr_results[doc_name].get("tags", [])
                
                true_has_tag = 1 if tag in true_tags else 0
                pred_has_tag = 1 if tag in pred_tags else 0
                
                y_true.append(true_has_tag)
                y_pred.append(pred_has_tag)
                
                if true_has_tag != pred_has_tag:
                    print(f"   📄 {doc_name}: GT={true_has_tag}, PRED={pred_has_tag}")
                    print(f"      GT tags: {true_tags}")
                    print(f"      OCR tags: {pred_tags}")
        
        if y_true and y_pred:
            # Calculate precision, recall, F1 for this tag
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            tag_results[tag] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': sum(y_true)  # Number of true positives in ground truth
            }
            
            print(f"🏷️  {tag}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            print(f"   F1-Score:  {f1:.3f}")
            print(f"   Support:   {sum(y_true)} documents")
            print()
    
    # Calculate macro-averaged F1 score
    if tag_results:
        macro_f1 = sum(result['f1'] for result in tag_results.values()) / len(tag_results)
        print(f"📊 MACRO-AVERAGED TAG F1-SCORE: {macro_f1:.3f}")
        print()
    
    return tag_results

def main():
    """Run F1-score evaluation"""
    
    print("🎯 F1-SCORE EVALUATION FOR DOCUMENT CLASSIFICATION")
    print("=" * 60)
    print()
    
    try:
        # Load data
        ocr_results, ground_truth = load_classification_data()
        
        print(f"📁 Loaded {len(ocr_results)} OCR results")
        print(f"📁 Loaded {len(ground_truth)} ground truth files")
        print()
        
        # Find matching documents
        matching_docs = set(ocr_results.keys()) & set(ground_truth.keys())
        print(f"📊 Found {len(matching_docs)} matching documents for evaluation")
        print()
        
        if not matching_docs:
            print("❌ No matching documents found. Cannot perform evaluation.")
            return
        
        # Evaluate categories
        cat_accuracy, y_true_cat, y_pred_cat = evaluate_categories(ocr_results, ground_truth)
        
        # Evaluate tags
        tag_results = evaluate_tags(ocr_results, ground_truth)
        
        # Save results
        evaluation_results = {
            "category_accuracy": cat_accuracy,
            "tag_results": tag_results,
            "evaluation_summary": {
                "total_documents": len(matching_docs),
                "category_accuracy": cat_accuracy,
                "macro_avg_tag_f1": sum(result['f1'] for result in tag_results.values()) / len(tag_results) if tag_results else 0.0
            }
        }
        
        with open("f1_evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        print("💾 Results saved to: f1_evaluation_results.json")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")

if __name__ == "__main__":
    main()
