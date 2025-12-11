#!/usr/bin/env python
"""
Model Training Script
Standalone script to train ML models for the Financial Intelligence Portal.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.classifier import FinancialDataClassifier
from src.anomaly_detector import AnomalyDetector
from config.config import DevelopmentConfig


def train_classifier():
    """Train the GL account classification model."""
    print("=" * 60)
    print("Training GL Account Classification Model")
    print("=" * 60)
    
    # Load training data
    data_loader = DataLoader(DevelopmentConfig.DATA_DIR)
    training_data = data_loader.load_training_data()
    
    print(f"Training data loaded: {len(training_data)} samples")
    print(f"Columns: {list(training_data.columns)}")
    print()
    
    # Initialize and train classifier
    model_path = DevelopmentConfig.MODELS_DIR / 'classifier_model.pkl'
    classifier = FinancialDataClassifier(model_path=model_path)
    
    classifier.train(training_data)
    
    print()
    print("Testing classifier with sample inputs...")
    
    # Test predictions
    test_cases = [
        ("Direct Materials - Steel Components for Assembly", "Grainger Industrial Supply"),
        ("Travel Expenses - Sales Conference Q4", "TravelPerk Business Travel"),
        ("Strategic Consulting - Process Optimization", "McKinsey & Company"),
    ]
    
    for description, vendor in test_cases:
        result = classifier.predict(description, vendor)
        print(f"\nInput: '{description[:40]}...' | Vendor: '{vendor}'")
        print(f"  Predicted Account: {result['suggested_account']}")
        print(f"  Confidence: {result['confidence']:.2%}")


def train_anomaly_detector():
    """Train the anomaly detection model."""
    print()
    print("=" * 60)
    print("Training Anomaly Detection Model")
    print("=" * 60)
    
    # Load journal entries
    data_loader = DataLoader(DevelopmentConfig.DATA_DIR)
    journal_entries = data_loader.load_journal_entries()
    
    print(f"Journal entries loaded: {len(journal_entries)} transactions")
    print(f"Amount range: ${journal_entries['amount_local_currency'].min():,.2f} - ${journal_entries['amount_local_currency'].max():,.2f}")
    print()
    
    # Initialize and train anomaly detector
    model_path = DevelopmentConfig.MODELS_DIR / 'anomaly_detector.pkl'
    detector = AnomalyDetector(model_path=model_path)
    
    detector.train(journal_entries)
    
    print()
    print("Testing anomaly detector with sample transactions...")
    
    # Test detections
    test_transactions = [
        {
            'amount_local_currency': 5000,
            'description': 'Normal office supplies purchase',
            'source_erp': 'SAP_ECC',
            'vendor_name': 'Office Depot'
        },
        {
            'amount_local_currency': 150000,
            'description': 'Large equipment purchase',
            'source_erp': 'SAP_ECC',
            'vendor_name': 'Caterpillar'
        },
        {
            'amount_local_currency': 50000,
            'description': 'Strategic Consulting',
            'source_erp': 'JDE',
            'vendor_name': 'McKinsey & Company'
        },
        {
            'amount_local_currency': 2500,
            'description': 'Misk Exp',
            'source_erp': 'ISERIES',
            'vendor_name': 'Unknown Vendor'
        }
    ]
    
    for tx in test_transactions:
        result = detector.detect(tx)
        print(f"\nTransaction: ${tx['amount_local_currency']:,.2f} - '{tx['description']}'")
        print(f"  Is Anomaly: {result['is_anomaly']}")
        print(f"  Score: {result['anomaly_score']:.2f}")
        print(f"  Reason: {result['anomaly_reason']}")


def main():
    """Run full model training pipeline."""
    print()
    print("#" * 60)
    print("#  TMHNA Financial Intelligence - Model Training")
    print("#" * 60)
    print()
    
    # Ensure models directory exists
    DevelopmentConfig.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train models
    train_classifier()
    train_anomaly_detector()
    
    print()
    print("=" * 60)
    print("Model Training Complete!")
    print("=" * 60)
    print(f"Models saved to: {DevelopmentConfig.MODELS_DIR}")


if __name__ == '__main__':
    main()

