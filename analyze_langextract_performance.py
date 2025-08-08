#!/usr/bin/env python3
"""
Analyze F1 scores by category and tag for LangExtract evaluation
"""

import json
import os
import glob
from collections import defaultdict, Counter

def analyze_by_category_and_tag():
    """Analyze F1 scores by document category and tags"""
    
    # Load all classification results
    classification_files = glob.glob('ocr_results/*_classification.json')
    
    # Track performance by category and tag
    category_performance = defaultdict(list)
    tag_performance = defaultdict(list)
    
    # Category and tag distributions
    category_distribution = Counter()
    tag_distribution = Counter()
    
    print("🔍 ANALYZING LANGEXTRACT PERFORMANCE BY CATEGORY & TAG")
    print("=" * 60)
    print()
    
    for classification_file in classification_files:
        # Get base name
        base_name = os.path.basename(classification_file).replace('_classification.json', '')
        
        # Load classification result
        with open(classification_file, 'r') as f:
            classification = json.load(f)
        
        category = classification['category']
        tags = classification['tags']
        
        # Update distributions
        category_distribution[category] += 1
        for tag in tags:
            tag_distribution[tag] += 1
        
        # Find corresponding F1 scores from evaluation report
        with open('f1_evaluation_report.json', 'r') as f:
            f1_report = json.load(f)
        
        # Find this file's F1 scores
        for result in f1_report['individual_results']:
            if base_name in result['file']:
                word_f1 = result['word_level']['f1']
                char_f1 = result['character_level']['f1']
                
                # Add to category performance
                category_performance[category].append({
                    'file': base_name,
                    'word_f1': word_f1,
                    'char_f1': char_f1,
                    'tags': tags
                })
                
                # Add to tag performance
                for tag in tags:
                    tag_performance[tag].append({
                        'file': base_name,
                        'word_f1': word_f1,
                        'char_f1': char_f1,
                        'category': category
                    })
                break
    
    # Print distributions
    print("📊 CATEGORY DISTRIBUTION:")
    print("-" * 25)
    for category, count in category_distribution.most_common():
        print(f"  {category}: {count} files")
    
    print("\n📊 TAG DISTRIBUTION:")
    print("-" * 20)
    for tag, count in tag_distribution.most_common():
        print(f"  {tag}: {count} files")
    
    # Calculate category performance
    print("\n📈 PERFORMANCE BY CATEGORY:")
    print("-" * 30)
    category_stats = {}
    for category, results in category_performance.items():
        word_f1s = [r['word_f1'] for r in results]
        char_f1s = [r['char_f1'] for r in results]
        
        avg_word_f1 = sum(word_f1s) / len(word_f1s)
        avg_char_f1 = sum(char_f1s) / len(char_f1s)
        
        category_stats[category] = {
            'count': len(results),
            'avg_word_f1': avg_word_f1,
            'avg_char_f1': avg_char_f1,
            'word_f1s': word_f1s,
            'char_f1s': char_f1s
        }
        
        print(f"  🏷️  {category} ({len(results)} files):")
        print(f"     Word F1: {avg_word_f1:.3f}")
        print(f"     Char F1: {avg_char_f1:.3f}")
    
    # Calculate tag performance
    print("\n🏷️  PERFORMANCE BY TAG:")
    print("-" * 25)
    tag_stats = {}
    for tag, results in tag_performance.items():
        word_f1s = [r['word_f1'] for r in results]
        char_f1s = [r['char_f1'] for r in results]
        
        avg_word_f1 = sum(word_f1s) / len(word_f1s)
        avg_char_f1 = sum(char_f1s) / len(char_f1s)
        
        tag_stats[tag] = {
            'count': len(results),
            'avg_word_f1': avg_word_f1,
            'avg_char_f1': avg_char_f1,
            'word_f1s': word_f1s,
            'char_f1s': char_f1s
        }
        
        print(f"  🔖 {tag} ({len(results)} files):")
        print(f"     Word F1: {avg_word_f1:.3f}")
        print(f"     Char F1: {avg_char_f1:.3f}")
    
    # Save detailed analysis
    analysis_report = {
        'distributions': {
            'categories': dict(category_distribution),
            'tags': dict(tag_distribution)
        },
        'category_performance': category_stats,
        'tag_performance': tag_stats,
        'individual_classifications': {}
    }
    
    # Add individual file details
    for classification_file in classification_files:
        base_name = os.path.basename(classification_file).replace('_classification.json', '')
        
        with open(classification_file, 'r') as f:
            classification = json.load(f)
        
        analysis_report['individual_classifications'][base_name] = classification
    
    # Save analysis report
    with open('langextract_analysis_report.json', 'w') as f:
        json.dump(analysis_report, f, indent=2)
    
    print(f"\n📄 Detailed analysis saved to: langextract_analysis_report.json")
    
    # Summary insights
    print(f"\n🎯 KEY INSIGHTS:")
    print("-" * 15)
    
    # Best performing category
    best_category = max(category_stats.items(), key=lambda x: x[1]['avg_char_f1'])
    print(f"   🏆 Best Category: {best_category[0]} (Char F1: {best_category[1]['avg_char_f1']:.3f})")
    
    # Best performing tag
    best_tag = max(tag_stats.items(), key=lambda x: x[1]['avg_char_f1'])
    print(f"   🏆 Best Tag: {best_tag[0]} (Char F1: {best_tag[1]['avg_char_f1']:.3f})")
    
    # Tag complexity (files with multiple tags)
    multi_tag_files = []
    for base_name, classification in analysis_report['individual_classifications'].items():
        if len(classification['tags']) > 1:
            multi_tag_files.append((base_name, len(classification['tags'])))
    
    print(f"   📑 Multi-tag files: {len(multi_tag_files)}/{len(classification_files)} files")
    if multi_tag_files:
        avg_tags = sum(count for _, count in multi_tag_files) / len(multi_tag_files)
        print(f"   📊 Average tags per multi-tag file: {avg_tags:.1f}")

if __name__ == "__main__":
    analyze_by_category_and_tag()
