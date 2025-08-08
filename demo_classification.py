#!/usr/bin/env python3
"""
Enhanced Demo Script for PaddleOCR POC with Document Classification

This script demonstrates the document classification capabilities 
using sample text without requiring full OCR processing.
"""

import json
from pathlib import Path
from document_classifier import DocumentClassifier


def demo_classification():
    """Demonstrate document classification with various sample texts."""
    
    print("=" * 60)
    print("PADDLEOCR POC - DOCUMENT CLASSIFICATION DEMO")
    print("=" * 60)
    
    # Initialize classifier
    print("Initializing Document Classifier...")
    classifier = DocumentClassifier()
    print("✓ Classifier ready\n")
    
    # Sample documents representing different categories
    sample_documents = [
        {
            "filename": "access_agreement_letter.pdf",
            "text": """
            Our Ref: T&D/841/01/0032
            
            Dear Sir/Madam,
            
            Re: Access Agreement for Electrical Installation
            
            Thank you for your letter dated 15th October 2024 regarding the access agreement 
            for electrical installation maintenance. We are pleased to confirm the access 
            arrangements for your team to enter the premises for routine inspection and 
            maintenance of the electrical systems.
            
            The agreement covers access authorization for fire services equipment and 
            ventilation system maintenance as outlined in the attached documentation.
            
            Yours sincerely,
            John Smith
            Senior Engineer
            """,
            "expected": {"category": "Letter", "tags": ["Agreement (Access)", "Agreement (Fire Services)"]}
        },
        {
            "filename": "maintenance_inspection_report.pdf", 
            "text": """
            Monthly Maintenance Record - October 2024
            
            Equipment: Fire Services System - Block A
            Inspection Date: 2024-10-20
            Inspector: Technical Team
            
            Routine maintenance and inspection completed for fire services equipment.
            All systems operational. Cleaning of fire detection sensors completed.
            Ventilation system checked and serviced.
            
            Next inspection scheduled: November 2024
            """,
            "expected": {"category": "Maintenance Record", "tags": ["Agreement (Fire Services)"]}
        },
        {
            "filename": "civil_defect_notice.pdf",
            "text": """
            Defect Notification Report
            
            Location: Main Building - Level 3
            Date Reported: 2024-10-18
            
            Civil defect identified during routine inspection:
            - Structural crack in concrete wall
            - Access pathway obstruction due to debris
            - Requires immediate attention
            
            Contractor notified for repair works.
            Access may be restricted until defect is resolved.
            """,
            "expected": {"category": "Unknown", "tags": ["Defect (Civil)", "Defect (Access)"]}
        },
        {
            "filename": "commissioning_checklist.pdf",
            "text": """
            System Commissioning Record
            
            Project: New HVAC Installation
            Commissioning Date: 2024-10-15
            
            Initial startup and commissioning of ventilation system completed.
            All tests passed according to specification requirements.
            System ready for operation.
            
            Commissioning engineer: Mike Wong
            """,
            "expected": {"category": "Commissioning Record", "tags": ["Agreement (Ventilation)"]}
        },
        {
            "filename": "cleaning_service_log.pdf",
            "text": """
            Cleaning Service Record - Weekly Report
            
            Week Ending: 2024-10-20
            Service Areas: All common areas and technical rooms
            
            Regular cleaning and sanitization completed.
            Special attention to ventilation intake areas.
            All cleaning supplies restocked.
            
            Next service: 2024-10-27
            """,
            "expected": {"category": "Cleaning", "tags": []}
        },
        {
            "filename": "slope_maintenance_notice.pdf",
            "text": """
            Slope Work Notification
            
            Location: External Slope Area - Section C
            Scheduled Work: Slope stabilization and drainage maintenance
            
            Slope work to commence on 2024-10-25.
            Access to the area will be restricted during maintenance.
            
            Contractor: ABC Slope Engineering Ltd.
            Expected completion: 2024-11-05
            """,
            "expected": {"category": "Slope Work", "tags": ["Defect (Access)"]}
        }
    ]
    
    # Process each sample document
    all_results = []
    
    print("Processing sample documents...\n")
    
    for i, doc in enumerate(sample_documents, 1):
        print(f"Document {i}: {doc['filename']}")
        print("-" * 40)
        
        # Classify the document
        result = classifier.classify_document(doc["text"], doc["filename"])
        all_results.append(result)
        
        # Display results
        print(f"Category: {result['category']} (confidence: {result['confidence']:.2f})")
        print(f"Tags: {', '.join(result['tags']) if result['tags'] else 'None'}")
        print(f"Keywords found: {', '.join(result['keywords_found'][:5])}{'...' if len(result['keywords_found']) > 5 else ''}")
        
        # Compare with expected (if available)
        expected = doc.get("expected", {})
        if expected:
            category_match = "✓" if result['category'] == expected['category'] else "✗"
            print(f"Expected category: {expected['category']} {category_match}")
            
            expected_tags = set(expected.get('tags', []))
            predicted_tags = set(result['tags'])
            tag_overlap = len(expected_tags & predicted_tags)
            tag_match = "✓" if tag_overlap > 0 or (not expected_tags and not predicted_tags) else "✗"
            print(f"Expected tags: {', '.join(expected['tags']) if expected['tags'] else 'None'} {tag_match}")
        
        print()
    
    # Generate summary statistics
    print("=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    
    summary = classifier.get_classification_summary(all_results)
    
    print(f"Total documents processed: {summary['total_documents']}")
    print(f"Average confidence: {summary['average_confidence']:.2f}")
    
    print(f"\nCategory distribution:")
    for category, count in summary['category_distribution'].items():
        print(f"  {category}: {count}")
    
    print(f"\nTag distribution:")
    for tag, count in summary['tag_distribution'].items():
        print(f"  {tag}: {count}")
    
    print(f"\nConfidence levels:")
    print(f"  High confidence (≥0.7): {summary['high_confidence_docs']}")
    print(f"  Medium confidence (0.3-0.7): {summary['medium_confidence_docs']}")
    print(f"  Low confidence (<0.3): {summary['low_confidence_docs']}")
    
    # Save demo results
    demo_output = {
        "demo_timestamp": "2024-10-20",
        "classification_results": all_results,
        "summary_statistics": summary
    }
    
    output_file = Path("demo_classification_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(demo_output, f, ensure_ascii=False, indent=2)
    
    print(f"\nDemo results saved to: {output_file}")
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return demo_output


def test_multilingual_classification():
    """Test classification with bilingual (English/Chinese) content."""
    
    print("\n" + "=" * 60)
    print("BILINGUAL CLASSIFICATION TEST")
    print("=" * 60)
    
    classifier = DocumentClassifier()
    
    bilingual_samples = [
        {
            "filename": "bilingual_agreement.pdf",
            "text": """
            通行協議 Access Agreement
            
            敬啟者 Dear Sir/Madam,
            
            關於消防服務設備的通行協議，我們確認以下安排：
            Regarding the access agreement for fire services equipment, we confirm the following arrangements:
            
            1. 定期檢查和維修 Regular inspection and maintenance
            2. 緊急通道保持暢通 Emergency access routes to remain clear
            
            此致 Yours sincerely
            """,
            "description": "Bilingual agreement document"
        },
        {
            "filename": "chinese_maintenance_report.pdf", 
            "text": """
            維修記錄 Maintenance Record
            
            設備：通風系統
            Equipment: Ventilation System
            
            檢查日期：2024年10月20日
            Inspection Date: 20 October 2024
            
            所有系統運作正常，清潔工作已完成
            All systems operational, cleaning work completed
            """,
            "description": "Bilingual maintenance record"
        }
    ]
    
    for sample in bilingual_samples:
        print(f"\nDocument: {sample['filename']}")
        print(f"Description: {sample['description']}")
        print("-" * 40)
        
        result = classifier.classify_document(sample["text"], sample["filename"])
        
        print(f"Category: {result['category']} (confidence: {result['confidence']:.2f})")
        print(f"Tags: {', '.join(result['tags']) if result['tags'] else 'None'}")
        print(f"Keywords found: {', '.join(result['keywords_found'])}")


if __name__ == "__main__":
    try:
        # Run main classification demo
        demo_results = demo_classification()
        
        # Run bilingual test
        test_multilingual_classification()
        
        print(f"\n✓ Demo completed successfully!")
        print(f"✓ Check 'demo_classification_results.json' for detailed results")
        
    except Exception as e:
        print(f"Error running demo: {str(e)}")
        print("Please ensure the document_classifier.py module is working correctly.")
