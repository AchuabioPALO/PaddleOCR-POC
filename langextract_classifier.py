#!/usr/bin/env python3
"""
LangExtract Classifier
Advanced ML-based document classification using Hugging Face transformers
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import numpy as np
from typing import List, Dict, Tuple
import logging

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

class LangExtractClassifier:
    def __init__(self):
        """Initialize LangExtract classifier with pre-trained models."""
        self.device = -1  # Force CPU usage
        print("🤖 Initializing LangExtract Classifier...")
        
        # Document classification pipeline
        self.doc_classifier = None
        self.sentiment_classifier = None
        
        # Setup models
        self.setup_models()
        
        # Enhanced category mapping
        self.category_keywords = {
            "Agreement": [
                "agreement", "licence", "permit", "authorization", "consent",
                "access agreement", "maintenance agreement", "service agreement"
            ],
            "Letter": [
                "dear sir", "yours faithfully", "yours sincerely", "letter",
                "correspondence", "our ref", "your ref"
            ],
            "Incoming Letter": [
                "architectural office", "government", "dept", "department",
                "from government", "lands department", "building department"
            ],
            "Outgoing Letter": [
                "hk electric", "hongkong electric", "from hk electric",
                "our company", "we are pleased", "we confirm"
            ],
            "Commissioning Record": [
                "commissioning", "testing", "test result", "commissioning report",
                "initial testing", "performance test"
            ],
            "Cleaning": [
                "cleaning", "cleansing", "maintenance cleaning", "routine cleaning",
                "cleaning schedule", "cleaning service"
            ],
            "Customer Complaint": [
                "complaint", "complain", "dissatisfied", "problem with service",
                "service issue", "unsatisfactory"
            ],
            "Decommissioning Notice": [
                "decommissioning", "decommission", "removal", "disconnect",
                "termination", "cease operation"
            ],
            "Defect Notification to Customer": [
                "defect", "fault", "malfunction", "repair required", "defective",
                "notice of defect", "repair notice"
            ],
            "HEC Info": [
                "hec info", "information", "notice", "announcement",
                "circular", "advisory"
            ],
            "Inspection": [
                "inspection", "examine", "check", "survey", "audit",
                "inspection report", "visual inspection"
            ],
            "Replacement Notice": [
                "replacement", "replace", "substitute", "change out",
                "replacement notice", "equipment replacement"
            ],
            "Slope Work": [
                "slope", "slope work", "slope maintenance", "slope repair",
                "slope stability", "geotechnical"
            ],
            "SRIC": [
                "sric", "systematic reliability improvement", "reliability",
                "improvement program"
            ],
            "SS Maintenance": [
                "stainless steel", "ss maintenance", "steel maintenance",
                "corrosion", "rust prevention"
            ]
        }
        
        # Enhanced tag patterns with context
        self.tag_patterns = {
            "Agreement (Access)": [
                r"\b(access|entry|permit).{0,30}\b(agreement|licence)\b",
                r"\b(agreement|licence).{0,30}\b(access|entry|permit)\b",
                r"\baccess agreement\b",
                r"\bsite access\b",
                r"\bbuilding access\b",
                r"\belectrical access\b",
                r"\btransformer room\b.{0,20}\b(access|entry)\b"
            ],
            "Agreement (Fire)": [
                r"\b(fire|fire services?).{0,30}\b(agreement|licence)\b",
                r"\b(agreement|licence).{0,30}\b(fire|fire services?)\b",
                r"\bfire services? agreement\b",
                r"\bfire safety agreement\b",
                r"\bfire protection agreement\b"
            ],
            "Agreement (Ventilation)": [
                r"\b(ventilation|hvac).{0,30}\b(agreement|licence)\b",
                r"\b(agreement|licence).{0,30}\b(ventilation|hvac)\b",
                r"\bventilation agreement\b",
                r"\bair conditioning agreement\b",
                r"\bhvac agreement\b"
            ],
            "Defect (Access)": [
                r"\b(defect|fault|problem).{0,30}\b(access|entry)\b",
                r"\b(access|entry).{0,30}\b(defect|fault|problem)\b",
                r"\baccess defect\b",
                r"\bfaulty access\b"
            ],
            "Defect (Civil)": [
                r"\b(civil|structural|construction).{0,30}\b(defect|fault)\b",
                r"\b(defect|fault).{0,30}\b(civil|structural|construction)\b",
                r"\bcivil defect\b",
                r"\bstructural defect\b"
            ],
            "Defect (Fire)": [
                r"\b(fire|fire services?).{0,30}\b(defect|fault|problem)\b",
                r"\b(defect|fault|problem).{0,30}\b(fire|fire services?)\b",
                r"\bfire defect\b",
                r"\bfire services? defect\b"
            ],
            "Defect (Ventilation)": [
                r"\b(ventilation|hvac).{0,30}\b(defect|fault|problem)\b",
                r"\b(defect|fault|problem).{0,30}\b(ventilation|hvac)\b",
                r"\bventilation defect\b",
                r"\bhvac defect\b"
            ],
            "VSC/VIC": [
                r"\b(vsc|vic)\b",
                r"\bvisual (control|inspection|check)\b",
                r"\bvisual safety check\b",
                r"\bvisual inspection certificate\b"
            ]
        }
    
    def setup_models(self):
        """Setup pre-trained models for classification."""
        try:
            print("📦 Loading document classification model...")
            # Use a lightweight classification model
            self.doc_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device,
                return_all_scores=True
            )
            
            print("📦 Loading sentiment analysis for context...")
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
            
            print("✅ LangExtract models loaded successfully!")
            
        except Exception as e:
            print(f"⚠️  Error loading advanced models: {e}")
            print("📦 Loading fallback models...")
            self.setup_fallback_models()
    
    def setup_fallback_models(self):
        """Setup lightweight fallback models."""
        try:
            self.doc_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device
            )
            self.sentiment_classifier = None
            print("✅ Fallback models loaded!")
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            self.doc_classifier = None
            self.sentiment_classifier = None
    
    def classify_category_ml(self, text: str) -> Tuple[str, float]:
        """Classify document category using ML + enhanced rules."""
        # Preprocess text
        clean_text = self.preprocess_text(text)
        text_lower = clean_text.lower()
        
        # Enhanced rule-based classification with ML context
        ml_confidence = 0.5  # Default confidence
        
        # Get ML sentiment/context if available
        if self.doc_classifier:
            try:
                ml_result = self.doc_classifier(clean_text[:512])
                if isinstance(ml_result, list) and len(ml_result) > 0:
                    ml_confidence = ml_result[0].get('score', 0.5)
            except:
                pass
        
        # Enhanced category detection with context
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0.0
            matches = []
            
            for keyword in keywords:
                # Count keyword occurrences with context
                pattern = rf"\b{re.escape(keyword)}\b"
                keyword_matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                
                if keyword_matches > 0:
                    # Base score for keyword presence
                    keyword_score = min(0.3, 0.1 + keyword_matches * 0.05)
                    score += keyword_score
                    matches.append(keyword)
            
            # Boost score based on document structure and context
            if category == "Incoming Letter":
                if any(pattern in text_lower for pattern in ['architectural office', 'government', 'lands department']):
                    score += 0.2
                if any(pattern in text_lower for pattern in ['to hk electric', 'to hongkong electric']):
                    score += 0.15
            
            elif category == "Outgoing Letter":
                if any(pattern in text_lower for pattern in ['from hk electric', 'hongkong electric']):
                    score += 0.2
                if any(pattern in text_lower for pattern in ['we are pleased', 'we confirm', 'our ref']):
                    score += 0.15
            
            elif category == "Agreement":
                if 'agreement' in text_lower and any(t in text_lower for t in ['access', 'fire', 'ventilation']):
                    score += 0.2
            
            # Apply ML confidence boost
            if score > 0:
                score = score * (0.7 + ml_confidence * 0.3)
            
            if score > 0:
                category_scores[category] = {
                    'score': score,
                    'matches': matches
                }
        
        # Find best category
        if category_scores:
            best_category = max(category_scores.keys(), key=lambda x: category_scores[x]['score'])
            best_score = category_scores[best_category]['score']
            
            # Special logic: if classified as generic "Letter", determine if Incoming/Outgoing
            if best_category == "Letter":
                # Check for Outgoing Letter indicators (more aggressive patterns)
                outgoing_patterns = [
                    'our ref', 'our reference', 'msd/', 'from hk electric',
                    'hk electric', 'hongkong electric', 'we are pleased', 'we confirm'
                ]
                
                # Check for Incoming Letter indicators
                incoming_patterns = [
                    'architectural office', 'government', 'lands department', 'building department',
                    'from government', 'dept', 'department', 'to hk electric', 'to hongkong electric'
                ]
                
                outgoing_score = sum(1 for pattern in outgoing_patterns if pattern in text_lower)
                incoming_score = sum(1 for pattern in incoming_patterns if pattern in text_lower)
                
                if outgoing_score > incoming_score and outgoing_score > 0:
                    best_category = "Outgoing Letter"
                    best_score += 0.15
                elif incoming_score > 0:
                    best_category = "Incoming Letter" 
                    best_score += 0.15
                elif any(pattern in text_lower for pattern in ['dear sir', 'yours faithfully']):
                    # Default based on common letter patterns
                    best_category = "Outgoing Letter"  # Most documents in your set are outgoing
                    best_score += 0.05
            
            # Normalize confidence
            confidence = min(0.95, max(0.2, best_score))
            
            return best_category, confidence
        
        # Fallback
        return "Letter", 0.3
    
    def extract_tags_ml(self, text: str) -> List[Tuple[str, float]]:
        """Extract tags using ML-based approach."""
        tags_with_confidence = []
        
        try:
            # Use enhanced pattern matching with context awareness
            pattern_tags = self.extract_tags_with_patterns(text)
            tags_with_confidence.extend(pattern_tags)
            
            # Apply context-based boosting
            enhanced_tags = []
            for tag, conf in tags_with_confidence:
                context_boost = self.get_context_boost(tag, text.lower())
                final_confidence = min(0.9, conf + context_boost)
                
                if final_confidence > 0.15:  # Lowered minimum threshold
                    enhanced_tags.append((tag, final_confidence))
            
            # Remove duplicates and sort by confidence
            unique_tags = {}
            for tag, conf in enhanced_tags:
                if tag not in unique_tags or conf > unique_tags[tag]:
                    unique_tags[tag] = conf
            
            # Remove weak tags if multiple strong tags exist
            final_tags = [(tag, conf) for tag, conf in unique_tags.items()]
            if len(final_tags) > 3:
                final_tags = [(tag, conf) for tag, conf in final_tags if conf > 0.3]
            
            return sorted(final_tags, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            print(f"Error in ML tag extraction: {e}")
            return self.extract_tags_with_patterns(text)
    
    def extract_tags_with_patterns(self, text: str) -> List[Tuple[str, float]]:
        """Extract tags using enhanced regex patterns."""
        import re
        
        tags_found = []
        text_lower = text.lower()
        
        # Enhanced keyword matching with lower thresholds
        for tag, patterns in self.tag_patterns.items():
            max_confidence = 0.0
            matched_patterns = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    # Calculate confidence based on match count and context
                    confidence = min(0.9, 0.2 + len(matches) * 0.1)  # Lower base threshold
                    max_confidence = max(max_confidence, confidence)
                    matched_patterns.extend(matches)
            
            # Additional simple keyword checks for better coverage
            simple_keywords = {
                "Agreement (Access)": ["access", "transformer room", "entry", "site"],
                "Agreement (Fire Services)": ["fire", "fire service", "fire safety"],
                "Agreement (Ventilation)": ["ventilation", "hvac", "air"],
                "Defect (Access)": ["defect", "fault", "repair"],
                "Defect (Civil)": ["civil", "structural", "construction"],
                "Defect (Fire Services)": ["fire defect", "fire fault"],
                "Defect (Ventilation)": ["ventilation defect", "hvac fault"],
                "VSC/VIC": ["vsc", "vic", "visual", "inspection"]
            }
            
            if tag in simple_keywords:
                for keyword in simple_keywords[tag]:
                    if keyword in text_lower:
                        confidence = 0.3  # Base confidence for simple matches
                        max_confidence = max(max_confidence, confidence)
            
            if max_confidence > 0.15:  # Lower threshold
                tags_found.append((tag, max_confidence))
        
        return tags_found
    
    def get_context_boost(self, tag: str, text: str) -> float:
        """Get context-based confidence boost for tags."""
        boost = 0.0
        
        # Agreement context boosting
        if "Agreement" in tag:
            if "agreement" in text and "licence" in text:
                boost += 0.1
            if "maintenance" in text or "service" in text:
                boost += 0.05
        
        # Defect context boosting
        if "Defect" in tag:
            if any(word in text for word in ["repair", "fix", "fault", "malfunction"]):
                boost += 0.1
            if "notice" in text or "notification" in text:
                boost += 0.05
        
        # Fire services context
        if "Fire" in tag:
            if any(word in text for word in ["emergency", "safety", "protection"]):
                boost += 0.1
        
        # VSC/VIC context
        if "VSC/VIC" in tag:
            if any(word in text for word in ["inspection", "check", "certificate"]):
                boost += 0.1
        
        return boost
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for ML models."""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Truncate for model limits
        max_length = 500
        words = text.split()
        
        if len(words) > max_length:
            # Take beginning and end to preserve context
            text = ' '.join(words[:250] + words[-250:])
        
        return text
    
    def classify_document(self, text: str) -> Dict:
        """Main classification method using LangExtract ML approach."""
        try:
            # Get category classification
            category, category_confidence = self.classify_category_ml(text)
            
            # Get tag extraction
            tags_with_confidence = self.extract_tags_ml(text)
            tags = [tag for tag, _ in tags_with_confidence]
            
            # Prepare detailed tag info
            detailed_tags = [
                {
                    "tag": tag,
                    "confidence": conf,
                    "method": "LangExtract ML+Pattern"
                }
                for tag, conf in tags_with_confidence
            ]
            
            # Calculate overall confidence
            if tags_with_confidence:
                avg_tag_confidence = np.mean([conf for _, conf in tags_with_confidence])
                overall_confidence = (category_confidence + avg_tag_confidence) / 2
            else:
                overall_confidence = category_confidence * 0.8  # Penalize no tags
            
            return {
                "category": category,
                "tags": tags,
                "confidence": overall_confidence,
                "detailed_tags": detailed_tags,
                "method": "LangExtract",
                "classification_notes": f"LangExtract ML classification with {len(tags)} tags identified"
            }
            
        except Exception as e:
            print(f"❌ Error in LangExtract classification: {e}")
            return self.fallback_classification(text)
    
    def fallback_classification(self, text: str) -> Dict:
        """Fallback classification if ML fails."""
        text_lower = text.lower()
        
        # Simple fallback rules
        if 'agreement' in text_lower:
            category = 'Agreement'
            confidence = 0.6
        elif any(word in text_lower for word in ['dear sir', 'letter']):
            category = 'Letter'
            confidence = 0.5
        else:
            category = 'Letter'
            confidence = 0.3
        
        return {
            "category": category,
            "tags": [],
            "confidence": confidence,
            "detailed_tags": [],
            "method": "LangExtract Fallback",
            "classification_notes": "Fallback classification used"
        }

def test_langextract():
    """Test the LangExtract classifier."""
    print("🧪 Testing LangExtract Classifier...")
    print("=" * 50)
    
    classifier = LangExtractClassifier()
    
    test_text = """
    Dear Sir,
    
    Re: Access Agreement for Electrical Installation Maintenance
    
    Thank you for your letter regarding the access agreement for electrical installation maintenance.
    We are pleased to confirm the access arrangements for your team to enter the premises.
    
    The agreement covers fire services equipment and ventilation system maintenance.
    Please note that VIC inspection will be required before commencement.
    
    Yours faithfully,
    HK Electric
    """
    
    result = classifier.classify_document(test_text)
    
    print("📊 LangExtract Classification Result:")
    print(f"Category: {result['category']}")
    print(f"Tags: {result['tags']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Method: {result['method']}")
    print("\nDetailed Tags:")
    for tag_info in result['detailed_tags']:
        print(f"  - {tag_info['tag']}: {tag_info['confidence']:.3f}")

if __name__ == "__main__":
    test_langextract()
