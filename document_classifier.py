#!/usr/bin/env python3
"""
Document Classification Module for PaddleOCR POC

This module provides document classification functionality to categorize documents
into predefined types and assign relevant tags based on extracted text content.
"""

import re
import json
from typing import Dict, List, Tuple
from pathlib import Path


class DocumentClassifier:
    def __init__(self):
        """Initialize the document classifier with keyword mappings."""
        
        # Document categories based on your actual folder structure
        self.category_keywords = {
            # Main Categories from your PDFs folder
            "Agreement": [
                "agreement", "licence", "permit", "authorization", "consent",
                "access agreement", "fire services agreement", "ventilation agreement",
                "協議", "許可證", "同意書", "授權書"
            ],
            "Cleaning": [
                "cleaning", "clean", "wash", "sanitize", "housekeeping",
                "cleaning record", "cleaning report", "清潔", "清洗", "清理"
            ],
            "Commissioning Record": [
                "commissioning", "commission", "initial test", "startup", "testing",
                "commissioning record", "commissioning report", "test certificate",
                "投產", "投運", "啟動", "調試記錄", "測試"
            ],
            "Customer Complaint": [
                "complaint", "complain", "dissatisfaction", "grievance",
                "customer complaint", "complaint record", "投訴", "客戶投訴"
            ],
            "Decommissioning Notice": [
                "decommissioning", "decommission", "removal", "dismantling",
                "decommissioning notice", "除役", "拆除"
            ],
            "Defect Notification to Customer": [
                "defect notification", "defect", "fault", "problem", "issue", 
                "malfunction", "fault report", "defect to customer", "缺陷", "故障"
            ],
            "HEC Info": [
                "hec info", "hec", "hong kong electric", "safety rules", "safety notice",
                "sric", "safety regulations", "港燈", "安全規則"
            ],
            "Inspection": [
                "inspection", "inspect", "examination", "check", "survey", "review",
                "inspection record", "inspection report", "檢查", "檢驗", "巡查"
            ],
            "Letter": [
                "letter", "correspondence", "dear sir", "dear madam", "yours sincerely",
                "yours faithfully", "敬啟者", "此致", "來函", "覆函"
            ],
            "Incoming Letter": [
                "architectural office", "public works department", "government", 
                "buildings department", "lands department", "to hk electric",
                "to hongkong electric", "from government", "official correspondence",
                "government reference", "pwd", "bd ref", "ld ref", "government office",
                "contractor", "consultant", "external party", "third party",
                "敬啟者", "政府部門", "建築署", "工務局"
            ],
            "Outgoing Letter": [
                "our ref", "our reference", "msd/", "from hk electric", "hk electric",
                "engineering department", "customer services", "港燈", "本公司",
                "despatched", "file ref", "file no"
            ],
            "Replacement Notice": [
                "replacement", "replace", "renewal", "substitution",
                "replacement notice", "更換", "替換"
            ],
            "Slope Work": [
                "slope work", "slope", "gradient", "slope maintenance",
                "slope repair", "earthwork", "邊坡", "斜坡工程"
            ],
            "SRIC": [
                "sric", "safety rules in chinese", "safety information", "safety card",
                "安全資訊", "安全卡", "安全規則"
            ],
            "SS Maintenance": [
                "ss maintenance", "stainless steel maintenance", "ss", "stainless",
                "stainless steel", "不銹鋼維修", "不銹鋼保養"
            ]
        }
        
        # Document tags with their identifying keywords
        self.tag_keywords = {
            "Agreement (Access)": [
                "agreement", "access agreement", "access", "entry", "entry permit",
                "access permit", "access authorization", "site access", "building access",
                "electrical access", "transformer room", "supply", "installation",
                "electrical supply", "electrical installation", "building", "premises",
                "通行協議", "進入協議", "通行許可"
            ],
            "Agreement (Fire Services)": [
                "fire services", "fire agreement", "fire safety", "fire protection",
                "fire services agreement", "fire system", "消防協議", "消防服務"
            ],
            "Agreement (Ventilation)": [
                "ventilation", "ventilation agreement", "air flow", "hvac",
                "ventilation services", "air conditioning", "通風協議", "通風系統"
            ],
            "Defect (Access)": [
                "defect", "access defect", "access problem", "access issue",
                "entry defect", "通行缺陷", "進入問題"
            ],
            "Defect (Civil)": [
                "civil defect", "structural defect", "construction defect",
                "civil problem", "土木缺陷", "結構缺陷"
            ],
            "Defect (Fire Services)": [
                "fire defect", "fire services defect", "fire safety defect",
                "fire protection defect", "消防缺陷", "消防問題"
            ],
            "Defect (Ventilation)": [
                "ventilation defect", "hvac defect", "air flow defect",
                "ventilation problem", "通風缺陷", "通風問題"
            ],
            "VSC/VIC": [
                "vsc", "vic", "visual control", "visual inspection",
                "visual check", "可視控制", "可視檢查"
            ]
        }
        
        # Confidence thresholds (lowered for better detection)
        self.category_threshold = 0.15  # Lowered from 0.2
        self.tag_threshold = 0.10       # Lowered from 0.15 - very low to catch more tags
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for classification."""
        if not text:
            return ""
        
        # Convert to lowercase and normalize whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation that might be important
        text = re.sub(r'[^\w\s\.\,\:\;\(\)\-\/]', ' ', text)
        
        return text
    
    def calculate_keyword_score(self, text: str, keywords: List[str]) -> Tuple[float, List[str]]:
        """Calculate similarity score based on keyword matching."""
        if not text or not keywords:
            return 0.0, []
        
        text = self.preprocess_text(text)
        found_keywords = []
        total_score = 0.0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact phrase matching
            if keyword_lower in text:
                found_keywords.append(keyword)
                # Weight longer keywords more heavily
                weight = len(keyword_lower.split()) * 0.3 + 0.7
                total_score += weight
            
            # Partial word matching for compound keywords
            elif len(keyword_lower.split()) > 1:
                words = keyword_lower.split()
                word_matches = sum(1 for word in words if word in text)
                if word_matches > 0:
                    partial_score = (word_matches / len(words)) * 0.5
                    total_score += partial_score
                    if partial_score > 0.3:  # Only add to found keywords if significant match
                        found_keywords.append(keyword + " (partial)")
        
        # Normalize score
        if keywords:
            normalized_score = min(total_score / len(keywords), 1.0)
        else:
            normalized_score = 0.0
        
        return normalized_score, found_keywords
    
    def classify_category(self, text: str, filename: str = "") -> Dict:
        """Classify document into a category with enhanced logic that prioritizes document structure."""
        best_category = "Unknown"
        best_score = 0.0
        best_keywords = []
        
        # Combine text and filename for classification
        combined_text = f"{text} {filename}"
        text_lower = combined_text.lower()
        
        # PRIORITY 1: Check for incoming letter patterns first (most specific)
        # These patterns indicate government/external correspondence TO the company
        incoming_patterns = [
            "architectural office", "public works", "buildings department",
            "lands department", "government", "dear sir", "工務司署", "建築設計處"
        ]
        
        if any(pattern in text_lower for pattern in incoming_patterns):
            incoming_score, keywords = self.calculate_keyword_score(combined_text, self.category_keywords["Incoming Letter"])
            # Boost confidence for government correspondence patterns
            if any(gov_pattern in text_lower for gov_pattern in ["architectural office", "public works", "工務司署"]):
                incoming_score += 0.3  # Boost confidence for clear government patterns
            
            if incoming_score >= self.category_threshold:
                return {
                    "category": "Incoming Letter",
                    "confidence": min(incoming_score, 1.0),  # Cap at 1.0
                    "keywords_found": keywords
                }
        
        # PRIORITY 2: Check for outgoing letter patterns
        # These patterns indicate company correspondence FROM the company
        outgoing_patterns = [
            "our ref", "hk electric", "hongkong electric", "港燈", 
            "the hongkong electric", "planning engineer", "t&d/", "yours faithfully"
        ]
        
        if any(pattern in text_lower for pattern in outgoing_patterns):
            outgoing_score, keywords = self.calculate_keyword_score(combined_text, self.category_keywords["Outgoing Letter"])
            if outgoing_score >= self.category_threshold:
                return {
                    "category": "Outgoing Letter", 
                    "confidence": outgoing_score,
                    "keywords_found": keywords
                }
        
        # PRIORITY 3: Standard category matching for all other categories
        # But reduce weights for maintenance categories if it's clearly a letter
        is_letter_like = any(pattern in text_lower for pattern in ["dear sir", "dear madam", "yours sincerely", "yours faithfully"])
        
        for category, keywords in self.category_keywords.items():
            if category in ["Incoming Letter", "Outgoing Letter"]:
                continue  # Already handled above
                
            score, found_keywords = self.calculate_keyword_score(combined_text, keywords)
            
            # Reduce score for maintenance/technical categories if it looks like correspondence
            if is_letter_like and category in ["SS Maintenance", "Maintenance", "Cleaning", "Inspection"]:
                score *= 0.5  # Reduce technical category scores for letter-like documents
            
            if score > best_score and score >= self.category_threshold:
                best_category = category
                best_score = score
                best_keywords = found_keywords
        
        return {
            "category": best_category,
            "confidence": best_score,
            "keywords_found": best_keywords
        }
    
    def classify_tags(self, text: str, filename: str = "") -> List[Dict]:
        """Classify document tags."""
        tags = []
        combined_text = f"{text} {filename}"
        
        for tag, keywords in self.tag_keywords.items():
            score, found_keywords = self.calculate_keyword_score(combined_text, keywords)
            
            if score >= self.tag_threshold:
                tags.append({
                    "tag": tag,
                    "confidence": score,
                    "keywords_found": found_keywords
                })
        
        # Sort tags by confidence
        tags.sort(key=lambda x: x["confidence"], reverse=True)
        
        return tags
    
    def classify_document(self, text: str, filename: str = "") -> Dict:
        """
        Classify a document into category and tags.
        
        Args:
            text: Extracted text content from the document
            filename: Original filename for additional context
            
        Returns:
            Dict containing classification results
        """
        if not text:
            return {
                "category": "Unknown",
                "tags": [],
                "confidence": 0.0,
                "keywords_found": [],
                "classification_notes": "No text content available for classification"
            }
        
        # Classify category
        category_result = self.classify_category(text, filename)
        
        # Classify tags
        tag_results = self.classify_tags(text, filename)
        
        # Extract just the tag names for the main result
        tag_names = [tag["tag"] for tag in tag_results]
        
        # Combine all keywords found
        all_keywords = category_result["keywords_found"].copy()
        for tag in tag_results:
            all_keywords.extend(tag["keywords_found"])
        
        return {
            "category": category_result["category"],
            "tags": tag_names,
            "confidence": category_result["confidence"],
            "keywords_found": list(set(all_keywords)),  # Remove duplicates
            "detailed_tags": tag_results,  # Keep detailed tag info for analysis
            "classification_notes": f"Classified using {len(all_keywords)} keyword matches"
        }
    
    def get_classification_summary(self, classifications: List[Dict]) -> Dict:
        """Generate a summary of classification results."""
        if not classifications:
            return {"total_documents": 0}
        
        category_counts = {}
        tag_counts = {}
        confidence_scores = []
        
        for classification in classifications:
            # Count categories
            category = classification.get("category", "Unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count tags
            for tag in classification.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Collect confidence scores
            confidence = classification.get("confidence", 0.0)
            confidence_scores.append(confidence)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            "total_documents": len(classifications),
            "category_distribution": category_counts,
            "tag_distribution": tag_counts,
            "average_confidence": avg_confidence,
            "high_confidence_docs": sum(1 for c in confidence_scores if c >= 0.7),
            "medium_confidence_docs": sum(1 for c in confidence_scores if 0.3 <= c < 0.7),
            "low_confidence_docs": sum(1 for c in confidence_scores if c < 0.3)
        }


def main():
    """Test the document classifier with sample text."""
    classifier = DocumentClassifier()
    
    # Test samples
    test_documents = [
        {
            "filename": "841-A-0032-Letter_Agreement.pdf",
            "text": "Dear Sir/Madam, This is regarding the access agreement for ventilation services. Yours sincerely,"
        },
        {
            "filename": "maintenance_report.pdf", 
            "text": "Monthly maintenance record for cleaning and inspection of fire services equipment."
        },
        {
            "filename": "defect_notice.pdf",
            "text": "Civil defect notification regarding structural issues found during inspection."
        }
    ]
    
    print("Document Classification Test Results:")
    print("=" * 60)
    
    all_classifications = []
    
    for doc in test_documents:
        result = classifier.classify_document(doc["text"], doc["filename"])
        all_classifications.append(result)
        
        print(f"\nDocument: {doc['filename']}")
        print(f"Category: {result['category']} (confidence: {result['confidence']:.2f})")
        print(f"Tags: {', '.join(result['tags']) if result['tags'] else 'None'}")
        print(f"Keywords found: {', '.join(result['keywords_found'])}")
    
    # Generate summary
    summary = classifier.get_classification_summary(all_classifications)
    print("\n" + "=" * 60)
    print("Classification Summary:")
    print(f"Total documents: {summary['total_documents']}")
    print(f"Average confidence: {summary['average_confidence']:.2f}")
    print(f"Category distribution: {summary['category_distribution']}")
    print(f"Tag distribution: {summary['tag_distribution']}")


if __name__ == "__main__":
    main()
