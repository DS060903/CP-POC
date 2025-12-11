"""
Financial Data Classifier Module
ML-based classification of journal entries to unified Chart of Accounts.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class FinancialDataClassifier:
    """
    Machine Learning classifier for mapping journal entries
    to the unified TMHNA Chart of Accounts.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize classifier, loading pre-trained model if available.
        
        Args:
            model_path: Path to pickle file with trained model.
        """
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_trained = False
        
        # Try to load existing model
        if model_path and model_path.exists():
            self.load_model()
    
    def train(self, training_data: pd.DataFrame) -> None:
        """
        Train the classifier on labeled training data.
        
        Args:
            training_data: DataFrame with columns:
                - description: Transaction description text
                - vendor_name: Vendor name
                - true_tmhna_account: Target TMHNA account code
        """
        print("Training GL Account Classification Model...")
        
        # Combine description and vendor for richer features
        training_data = training_data.copy()
        training_data['combined_text'] = (
            training_data['description'].fillna('') + ' ' + 
            training_data['vendor_name'].fillna('')
        )
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Transform text to feature vectors
        X = self.vectorizer.fit_transform(training_data['combined_text'])
        
        # Encode target labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(training_data['true_tmhna_account'].astype(str))
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X, y)
        
        self.is_trained = True
        print(f"Model trained on {len(training_data)} samples with {len(self.label_encoder.classes_)} classes")
        
        # Save model to disk
        if self.model_path:
            self.save_model()
    
    def predict(self, description: str, vendor_name: str = '') -> Dict:
        """
        Predict TMHNA account for a journal entry.
        
        Args:
            description: Transaction description text.
            vendor_name: Optional vendor name for additional context.
            
        Returns:
            Dictionary with:
                - suggested_account: Predicted TMHNA account code
                - confidence: Prediction confidence (0-1)
                - top_3_candidates: List of top 3 predictions with confidences
        """
        if not self.is_trained or self.model is None:
            return self._fallback_prediction(description, vendor_name)
        
        # Combine text features
        combined_text = f"{description} {vendor_name}"
        
        # Vectorize input
        X = self.vectorizer.transform([combined_text])
        
        # Get prediction and probabilities
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get top 3 candidates
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_candidates = [
            {
                'account': str(self.label_encoder.classes_[idx]),
                'confidence': float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
        
        # Get predicted account
        suggested_account = str(self.label_encoder.classes_[prediction])
        confidence = float(max(probabilities))
        
        return {
            'suggested_account': suggested_account,
            'confidence': confidence,
            'top_3_candidates': top_3_candidates
        }
    
    def _fallback_prediction(self, description: str, vendor_name: str) -> Dict:
        """
        Rule-based fallback when model is not trained.
        
        Args:
            description: Transaction description.
            vendor_name: Vendor name.
            
        Returns:
            Prediction dictionary with lower confidence.
        """
        text = f"{description} {vendor_name}".lower()
        
        # Simple keyword-based rules
        rules = [
            (['material', 'steel', 'component', 'raw', 'fastener'], '5000', 0.7),
            (['labor', 'payroll', 'wage', 'salary'], '5100', 0.7),
            (['freight', 'shipping', 'inbound'], '5200', 0.7),
            (['travel', 't&e', 'conference', 'expense'], '6000', 0.7),
            (['utility', 'electric', 'energy', 'power'], '6100', 0.7),
            (['consulting', 'advisory', 'mckinsey'], '6200', 0.7),
            (['maintenance', 'repair', 'equipment'], '6300', 0.7),
            (['revenue', 'sales', 'sold', 'product'], '7000', 0.7),
        ]
        
        for keywords, account, base_confidence in rules:
            if any(kw in text for kw in keywords):
                return {
                    'suggested_account': account,
                    'confidence': base_confidence,
                    'top_3_candidates': [
                        {'account': account, 'confidence': base_confidence},
                        {'account': '6400', 'confidence': 0.15},
                        {'account': '5000', 'confidence': 0.15}
                    ]
                }
        
        # Default fallback
        return {
            'suggested_account': '6400',
            'confidence': 0.5,
            'top_3_candidates': [
                {'account': '6400', 'confidence': 0.5},
                {'account': '5000', 'confidence': 0.25},
                {'account': '6000', 'confidence': 0.25}
            ]
        }
    
    def save_model(self) -> None:
        """Save trained model to pickle file."""
        if not self.model_path:
            return
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """
        Load trained model from pickle file.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not self.model_path or not self.model_path.exists():
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

