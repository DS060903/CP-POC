"""
Financial Intelligence Engine - Backend Orchestrator
Coordinates data loading, ML classification, anomaly detection, and NLP processing.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import DevelopmentConfig
from src.data_loader import DataLoader
from src.classifier import FinancialDataClassifier
from src.anomaly_detector import AnomalyDetector
from src.nlp_processor import NLPProcessor


class FinancialIntelligenceEngine:
    """
    Main orchestrator class that coordinates all financial intelligence components.
    Provides unified interface for data processing, classification, and analysis.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Financial Intelligence Engine.
        Loads all components and ensures models are trained.
        
        Args:
            config: Configuration object. Defaults to DevelopmentConfig.
        """
        self.config = config or DevelopmentConfig
        
        print("Initializing Financial Intelligence Engine...")
        
        # Initialize components
        self.data_loader = DataLoader(self.config.DATA_DIR)
        self.nlp_processor = NLPProcessor()
        
        # Initialize ML models with paths
        models_dir = self.config.MODELS_DIR
        models_dir.mkdir(exist_ok=True)
        
        self.classifier = FinancialDataClassifier(
            model_path=models_dir / 'classifier_model.pkl'
        )
        self.anomaly_detector = AnomalyDetector(
            model_path=models_dir / 'anomaly_detector.pkl'
        )
        
        # Load data
        self.journal_entries = self.data_loader.load_journal_entries()
        self.coa_mapping = self.data_loader.load_coa_mapping()
        self.cost_centers = self.data_loader.load_cost_centers()
        self.training_data = self.data_loader.load_training_data()
        
        # Ensure models are trained
        self._ensure_models_trained()
        
        # Cache for processed entries
        self._processed_entries = None
        
        print("Financial Intelligence Engine initialized successfully!")
    
    def _ensure_models_trained(self) -> None:
        """
        Check if models are trained, train them if not.
        Called automatically during initialization.
        """
        # Check and train classifier
        if not self.classifier.is_trained:
            print("Classifier not trained, training now...")
            self.classifier.train(self.training_data)
        
        # Check and train anomaly detector
        if not self.anomaly_detector.is_trained:
            print("Anomaly detector not trained, training now...")
            self.anomaly_detector.train(self.journal_entries)
    
    def process_journal_entries(self, entries_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Process journal entries with ML classification and anomaly detection.
        
        Args:
            entries_df: Optional DataFrame to process. Uses cached data if None.
            
        Returns:
            DataFrame with original data plus ML enrichment columns.
        """
        if entries_df is None:
            entries_df = self.journal_entries.copy()
        
        # Process each entry
        processed_entries = []
        
        for _, row in entries_df.iterrows():
            entry = row.to_dict()
            
            # Get classification
            classification = self.classifier.predict(
                description=str(entry.get('description', '')),
                vendor_name=str(entry.get('vendor_name', ''))
            )
            
            # Get anomaly detection
            anomaly = self.anomaly_detector.detect(entry)
            
            # Get NLP analysis
            keyword_scores = self.nlp_processor.get_keyword_scores(
                str(entry.get('description', ''))
            )
            dominant_category = self.nlp_processor.get_dominant_category(
                str(entry.get('description', ''))
            )
            ambiguity_score = self.nlp_processor.detect_ambiguity(
                str(entry.get('description', ''))
            )
            
            # Enrich entry with ML results
            entry['suggested_tmhna_account'] = classification['suggested_account']
            entry['classification_confidence'] = classification['confidence']
            entry['top_3_suggestions'] = classification['top_3_candidates']
            entry['is_anomaly'] = anomaly['is_anomaly']
            entry['anomaly_score'] = anomaly['anomaly_score']
            entry['anomaly_reason'] = anomaly['anomaly_reason']
            entry['anomaly_severity'] = anomaly['severity']
            entry['dominant_keyword_category'] = dominant_category
            entry['ambiguity_score'] = ambiguity_score
            
            processed_entries.append(entry)
        
        self._processed_entries = pd.DataFrame(processed_entries)
        return self._processed_entries.copy()
    
    def get_processed_entries(self) -> pd.DataFrame:
        """
        Get processed entries, processing if not already done.
        
        Returns:
            DataFrame with all processed entries.
        """
        if self._processed_entries is None:
            return self.process_journal_entries()
        return self._processed_entries.copy()
    
    def get_drill_down_detail(self, entry_id: str) -> Dict:
        """
        Get detailed drill-down information for a specific entry.
        
        Args:
            entry_id: The journal entry ID.
            
        Returns:
            Dictionary with ERP detail and mapping explanation.
        """
        # Get ERP system detail
        erp_detail = self.data_loader.get_erp_detail(entry_id)
        
        # Get entry from processed data
        processed = self.get_processed_entries()
        entry_row = processed[processed['entry_id'] == entry_id]
        
        if entry_row.empty:
            return {
                'erp_detail': erp_detail,
                'mapping_explanation': {
                    'error': 'Entry not found in processed data'
                }
            }
        
        entry = entry_row.iloc[0].to_dict()
        description = str(entry.get('description', ''))
        vendor_name = str(entry.get('vendor_name', ''))
        
        # Generate mapping explanation
        keywords = self.nlp_processor.extract_keywords(description)
        keyword_scores = self.nlp_processor.get_keyword_scores(description)
        vendor_analysis = self.nlp_processor.analyze_vendor(vendor_name)
        
        # Get COA mapping info
        local_account = str(entry.get('local_account_code', ''))
        source_erp = str(entry.get('source_erp', ''))
        coa_info = self.data_loader.get_coa_by_legacy_code(local_account, source_erp)
        
        mapping_explanation = {
            'detected_keywords': keywords[:10],  # Top 10 keywords
            'keyword_category_scores': {
                k: round(v, 2) for k, v in sorted(
                    keyword_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            },
            'vendor_analysis': vendor_analysis,
            'legacy_account_lookup': coa_info,
            'mapping_logic': self._generate_mapping_logic(entry, coa_info),
            'confidence_factors': self._get_confidence_factors(entry)
        }
        
        return {
            'erp_detail': erp_detail,
            'mapping_explanation': mapping_explanation
        }
    
    def _generate_mapping_logic(self, entry: Dict, coa_info: Optional[Dict]) -> str:
        """
        Generate human-readable explanation of the mapping logic.
        
        Args:
            entry: Processed entry dictionary.
            coa_info: COA mapping information.
            
        Returns:
            String explanation of mapping decision.
        """
        parts = []
        
        # Explain keyword detection
        dominant_cat = entry.get('dominant_keyword_category', 'unknown')
        parts.append(f"1. Keywords suggest category: {dominant_cat}")
        
        # Explain vendor hint
        vendor_analysis = self.nlp_processor.analyze_vendor(
            str(entry.get('vendor_name', ''))
        )
        if vendor_analysis['category_hint'] != 'unknown':
            parts.append(
                f"2. Vendor '{entry.get('vendor_name')}' indicates "
                f"{vendor_analysis['category_hint']} ({vendor_analysis['confidence']} confidence)"
            )
        
        # Explain legacy account mapping
        if coa_info:
            parts.append(
                f"3. Legacy account {entry.get('local_account_code')} maps to "
                f"TMHNA {coa_info['tmhna_account_code']} ({coa_info['tmhna_account_name']})"
            )
        
        # Final suggestion
        parts.append(
            f"4. ML model suggests account {entry.get('suggested_tmhna_account')} "
            f"with {entry.get('classification_confidence', 0):.1%} confidence"
        )
        
        return " → ".join(parts)
    
    def _get_confidence_factors(self, entry: Dict) -> List[Dict]:
        """
        Get factors that influenced classification confidence.
        
        Args:
            entry: Processed entry dictionary.
            
        Returns:
            List of confidence factor dictionaries.
        """
        factors = []
        
        confidence = entry.get('classification_confidence', 0)
        ambiguity = entry.get('ambiguity_score', 0)
        
        # Description clarity
        if ambiguity < 0.3:
            factors.append({
                'factor': 'Description Clarity',
                'impact': 'Positive',
                'detail': 'Clear, descriptive text with identifiable keywords'
            })
        elif ambiguity > 0.5:
            factors.append({
                'factor': 'Description Clarity',
                'impact': 'Negative',
                'detail': 'Ambiguous or abbreviated description'
            })
        
        # Vendor recognition
        vendor_analysis = self.nlp_processor.analyze_vendor(
            str(entry.get('vendor_name', ''))
        )
        if vendor_analysis['confidence'] == 'high':
            factors.append({
                'factor': 'Vendor Recognition',
                'impact': 'Positive',
                'detail': f"Known vendor with strong category association"
            })
        
        # Amount range
        amount = float(entry.get('amount_local_currency', 0))
        if 1000 <= amount <= 50000:
            factors.append({
                'factor': 'Amount Range',
                'impact': 'Neutral',
                'detail': 'Transaction amount within typical range'
            })
        elif amount > 50000:
            factors.append({
                'factor': 'Amount Range',
                'impact': 'Caution',
                'detail': 'High-value transaction requiring review'
            })
        
        return factors
    
    def get_summary_dashboard(self) -> Dict:
        """
        Get summary statistics for the dashboard.
        
        Returns:
            Dictionary with dashboard metrics.
        """
        processed = self.get_processed_entries()
        
        # Basic counts
        total_entries = len(processed)
        anomalies_detected = len(processed[processed['is_anomaly'] == True])
        
        # Confidence stats
        avg_confidence = processed['classification_confidence'].mean()
        high_confidence_count = len(
            processed[processed['classification_confidence'] >= 0.85]
        )
        
        # Entries by account
        entries_by_account = processed['suggested_tmhna_account'].value_counts().to_dict()
        
        # High risk entries (anomaly score > 0.7)
        high_risk = processed[processed['anomaly_score'] >= 0.7]
        high_risk_entries = []
        
        for _, row in high_risk.iterrows():
            high_risk_entries.append({
                'entry_id': row['entry_id'],
                'source_erp': row['source_erp'],
                'description': row['description'][:50] + '...' if len(str(row['description'])) > 50 else row['description'],
                'amount': f"${row['amount_local_currency']:,.2f}",
                'anomaly_reason': row['anomaly_reason'],
                'anomaly_score': row['anomaly_score']
            })
        
        return {
            'total_entries': total_entries,
            'anomalies': anomalies_detected,
            'anomaly_percent': round(anomalies_detected / total_entries * 100, 1) if total_entries > 0 else 0,
            'average_confidence': avg_confidence,
            'high_confidence': high_confidence_count,
            'entries_by_account': entries_by_account,
            'high_risk_entries': high_risk_entries
        }
    
    def get_account_name(self, account_code: str) -> str:
        """
        Look up account name by code.
        
        Args:
            account_code: TMHNA account code.
            
        Returns:
            Account name or 'Unknown' if not found.
        """
        match = self.coa_mapping[
            self.coa_mapping['tmhna_account_code'].astype(str) == str(account_code)
        ]
        if not match.empty:
            return match.iloc[0]['tmhna_account_name']
        return 'Unknown'

