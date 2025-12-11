"""
NLP Processor Module
Extracts keywords and detects ambiguity in financial transaction descriptions.
"""
import re
from typing import List, Dict


class NLPProcessor:
    """
    Processes financial transaction text to extract keywords,
    categorize transactions, and detect ambiguity.
    """
    
    def __init__(self):
        """Initialize NLP processor with keyword categories."""
        # Define keyword categories for financial classification
        self.keyword_categories = {
            'direct_materials': [
                'material', 'materials', 'raw', 'component', 'components',
                'steel', 'parts', 'fastener', 'fasteners', 'hardware',
                'assembly', 'production', 'manufacturing', 'supply', 'supplies'
            ],
            'direct_labor': [
                'labor', 'payroll', 'wages', 'salary', 'workers',
                'employee', 'employees', 'staff', 'overtime', 'benefits'
            ],
            'utilities': [
                'utility', 'utilities', 'electric', 'electricity', 'power',
                'gas', 'water', 'energy', 'plant', 'facility'
            ],
            'travel_entertainment': [
                'travel', 'expense', 'expenses', 'conference', 'meeting',
                'hotel', 'flight', 'airfare', 'entertainment', 'meals',
                't&e', 'trip', 'business'
            ],
            'consulting': [
                'consulting', 'consultant', 'advisory', 'professional',
                'services', 'strategy', 'strategic', 'mckinsey', 'deloitte',
                'pwc', 'kpmg', 'engagement'
            ],
            'freight': [
                'freight', 'shipping', 'inbound', 'outbound', 'logistics',
                'delivery', 'carrier', 'fedex', 'ups', 'transport'
            ],
            'maintenance': [
                'maintenance', 'repair', 'repairs', 'equipment', 'fix',
                'replacement', 'service', 'upkeep', 'conveyor'
            ],
            'revenue': [
                'revenue', 'sales', 'sale', 'product', 'income',
                'sold', 'customer', 'order', 'contract'
            ]
        }
        
        # Common stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'it', 'its', 'q4', 'q1', 'q2', 'q3',
            'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'october', 'november', 'december', 'monthly'
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from transaction description.
        
        Args:
            text: Transaction description text.
            
        Returns:
            List of extracted keywords (lowercase, alphabetic only).
        """
        if not text:
            return []
        
        # Tokenize: split on non-alphanumeric characters
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short tokens
        keywords = [
            token for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return keywords
    
    def get_keyword_scores(self, text: str) -> Dict[str, float]:
        """
        Calculate relevance scores for each category based on keyword matches.
        
        Args:
            text: Transaction description text.
            
        Returns:
            Dictionary mapping category names to relevance scores (0-1).
        """
        if not text:
            return {cat: 0.0 for cat in self.keyword_categories}
        
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.keyword_categories.items():
            # Count how many category keywords appear in the text
            match_count = sum(1 for kw in keywords if kw in text_lower)
            # Normalize score (max 1.0)
            scores[category] = min(match_count / 3.0, 1.0)
        
        return scores
    
    def detect_ambiguity(self, text: str) -> float:
        """
        Detect how ambiguous a transaction description is.
        Higher scores indicate more ambiguity (harder to classify).
        
        Args:
            text: Transaction description text.
            
        Returns:
            Float between 0.0 (clear) and 1.0 (highly ambiguous).
        """
        if not text:
            return 1.0
        
        text_lower = text.lower().strip()
        
        # Very short descriptions are ambiguous
        if len(text_lower) < 10:
            return 0.8
        
        # Multiple dashes often indicate coded/abbreviated entries
        dash_count = text_lower.count('-')
        if dash_count > 2:
            return 0.8
        
        # Check for ambiguous terms
        ambiguous_terms = ['miscellaneous', 'misk', 'misc', 'other', 'adjustment', 'sundry', 'various']
        for term in ambiguous_terms:
            if term in text_lower:
                return 0.6
        
        # Check for very generic descriptions
        generic_terms = ['exp', 'expense', 'payment', 'charge']
        generic_count = sum(1 for term in generic_terms if term in text_lower)
        if generic_count > 0 and len(self.extract_keywords(text)) < 3:
            return 0.5
        
        # Clear, descriptive text
        return 0.1
    
    def get_dominant_category(self, text: str) -> str:
        """
        Determine the dominant keyword category for a text.
        
        Args:
            text: Transaction description text.
            
        Returns:
            Name of the category with highest score, or 'unknown'.
        """
        scores = self.get_keyword_scores(text)
        
        if not scores or max(scores.values()) == 0:
            return 'unknown'
        
        return max(scores, key=scores.get)
    
    def analyze_vendor(self, vendor_name: str) -> Dict[str, str]:
        """
        Analyze vendor name for classification hints.
        
        Args:
            vendor_name: Name of the vendor.
            
        Returns:
            Dictionary with vendor analysis results.
        """
        if not vendor_name:
            return {'category_hint': 'unknown', 'confidence': 'low'}
        
        vendor_lower = vendor_name.lower()
        
        # Known vendor mappings
        vendor_hints = {
            'grainger': ('direct_materials', 'high'),
            'fastenal': ('direct_materials', 'high'),
            'caterpillar': ('direct_materials', 'medium'),
            'adp': ('direct_labor', 'high'),
            'travelperk': ('travel_entertainment', 'high'),
            'duke energy': ('utilities', 'high'),
            'mckinsey': ('consulting', 'high'),
            'deloitte': ('consulting', 'high'),
            'fedex': ('freight', 'high'),
            'ups': ('freight', 'high'),
        }
        
        for vendor_key, (category, confidence) in vendor_hints.items():
            if vendor_key in vendor_lower:
                return {'category_hint': category, 'confidence': confidence}
        
        return {'category_hint': 'unknown', 'confidence': 'low'}

