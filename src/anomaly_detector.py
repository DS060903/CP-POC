"""
Anomaly Detector Module
Uses Isolation Forest to detect unusual financial transactions.
"""
import pickle
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd


class AnomalyDetector:
    """
    Detects anomalies in financial transactions using Isolation Forest
    and rule-based checks for known risk patterns.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize anomaly detector, loading pre-trained model if available.
        
        Args:
            model_path: Path to pickle file with trained model.
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Risk thresholds
        self.high_amount_threshold = 20000
        self.very_high_amount_threshold = 100000
        
        # Try to load existing model
        if model_path and model_path.exists():
            self.load_model()
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the anomaly detector on journal entries.
        
        Args:
            data: DataFrame with journal entries containing amount_local_currency.
        """
        print("Training Anomaly Detection Model...")
        
        # Prepare features
        features = self._prepare_features(data)
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100
        )
        self.model.fit(X_scaled)
        
        self.is_trained = True
        print(f"Anomaly detector trained on {len(data)} transactions")
        
        # Save model
        if self.model_path:
            self.save_model()
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for anomaly detection.
        
        Args:
            data: DataFrame with transaction data.
            
        Returns:
            NumPy array of features.
        """
        features = pd.DataFrame()
        
        # Amount features
        features['amount'] = data['amount_local_currency'].abs()
        features['amount_log'] = np.log1p(features['amount'])
        
        # Calculate z-score (if enough data)
        if len(data) > 1:
            mean_amount = features['amount'].mean()
            std_amount = features['amount'].std()
            if std_amount > 0:
                features['amount_zscore'] = (features['amount'] - mean_amount) / std_amount
            else:
                features['amount_zscore'] = 0
        else:
            features['amount_zscore'] = 0
        
        return features[['amount', 'amount_log', 'amount_zscore']].values
    
    def detect(self, transaction: Dict) -> Dict:
        """
        Detect if a transaction is anomalous.
        
        Args:
            transaction: Dictionary with transaction details including:
                - amount_local_currency: Transaction amount
                - description: Transaction description
                - source_erp: Source ERP system
                
        Returns:
            Dictionary with:
                - is_anomaly: Boolean indicating if anomalous
                - anomaly_score: Float score (0-1, higher = more anomalous)
                - anomaly_reason: Human-readable explanation
                - severity: 'High' or 'Low'
        """
        amount = abs(float(transaction.get('amount_local_currency', 0)))
        description = str(transaction.get('description', '')).lower()
        source_erp = str(transaction.get('source_erp', ''))
        vendor_name = str(transaction.get('vendor_name', '')).lower()
        
        anomaly_reasons = []
        severity = 'Low'
        base_score = 0.0
        
        # Rule-based checks
        
        # Check 1: Very high transaction amount
        if amount > self.very_high_amount_threshold:
            anomaly_reasons.append(f"Very high transaction amount (${amount:,.2f})")
            base_score = max(base_score, 0.9)
            severity = 'High'
        elif amount > self.high_amount_threshold:
            anomaly_reasons.append(f"Unusually high transaction amount (${amount:,.2f})")
            base_score = max(base_score, 0.7)
        
        # Check 2: Ambiguous description
        ambiguous_terms = ['misk', 'misc', 'miscellaneous', 'other', 'sundry', 'adjustment']
        if any(term in description for term in ambiguous_terms):
            anomaly_reasons.append("Ambiguous transaction description")
            base_score = max(base_score, 0.6)
        
        # Check 3: High-risk vendor categories
        high_risk_vendors = ['mckinsey', 'deloitte', 'pwc', 'kpmg', 'consulting']
        if any(vendor in vendor_name for vendor in high_risk_vendors):
            if amount > 30000:
                anomaly_reasons.append("High-value consulting engagement")
                base_score = max(base_score, 0.8)
                severity = 'High'
        
        # Check 4: Unknown or unusual ERP
        valid_erps = ['SAP_ECC', 'JDE', 'ISERIES']
        if source_erp not in valid_erps:
            anomaly_reasons.append(f"Unknown ERP system: {source_erp}")
            base_score = max(base_score, 0.5)
        
        # Check 5: Very short description
        if len(description) < 10:
            anomaly_reasons.append("Very brief transaction description")
            base_score = max(base_score, 0.4)
        
        # Use ML model if trained
        if self.is_trained and self.model is not None:
            try:
                features = np.array([[amount, np.log1p(amount), 0]])  # z-score approx 0
                features_scaled = self.scaler.transform(features)
                ml_prediction = self.model.predict(features_scaled)[0]
                ml_score = -self.model.score_samples(features_scaled)[0]
                
                # Normalize ML score to 0-1 range
                ml_score_normalized = min(max(ml_score / 0.5, 0), 1)
                
                if ml_prediction == -1:  # Anomaly detected by ML
                    base_score = max(base_score, ml_score_normalized)
                    if "ML model flagged" not in str(anomaly_reasons):
                        anomaly_reasons.append("ML model flagged as statistical outlier")
            except Exception:
                pass  # Fall back to rule-based only
        
        # Determine final result
        is_anomaly = base_score >= 0.5 or len(anomaly_reasons) > 0
        
        if not anomaly_reasons:
            anomaly_reason = "No anomalies detected"
        else:
            anomaly_reason = "; ".join(anomaly_reasons)
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': round(base_score, 2),
            'anomaly_reason': anomaly_reason,
            'severity': severity if is_anomaly else 'None'
        }
    
    def save_model(self) -> None:
        """Save trained model to pickle file."""
        if not self.model_path:
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Anomaly detector saved to {self.model_path}")
    
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
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            
            print(f"Anomaly detector loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading anomaly detector: {e}")
            return False

