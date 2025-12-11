"""
DataLoader Module
Handles loading and managing mock ERP data from CSV files.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import DevelopmentConfig


class DataLoader:
    """
    Loads and manages financial data from CSV files.
    Simulates data from three fragmented ERP systems: SAP, JDE, iSeries.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir: Path to data directory. Defaults to config DATA_DIR.
        """
        self.data_dir = data_dir or DevelopmentConfig.DATA_DIR
        self._journal_entries = None
        self._coa_mapping = None
        self._cost_centers = None
        self._training_data = None
    
    def load_journal_entries(self) -> pd.DataFrame:
        """
        Load journal entries from CSV file.
        
        Returns:
            DataFrame with journal entry data, parsing posting_date as datetime.
        """
        if self._journal_entries is None:
            csv_path = self.data_dir / 'journal_entries_raw.csv'
            self._journal_entries = pd.read_csv(csv_path, parse_dates=['posting_date'])
        return self._journal_entries.copy()
    
    def load_coa_mapping(self) -> pd.DataFrame:
        """
        Load unified Chart of Accounts mapping.
        Maps legacy GL codes to TMHNA standard accounts.
        
        Returns:
            DataFrame with COA mapping data.
        """
        if self._coa_mapping is None:
            csv_path = self.data_dir / 'chart_of_accounts.csv'
            self._coa_mapping = pd.read_csv(csv_path)
        return self._coa_mapping.copy()
    
    def load_cost_centers(self) -> pd.DataFrame:
        """
        Load cost center hierarchy.
        Contains brand, location, and function mappings.
        
        Returns:
            DataFrame with cost center data.
        """
        if self._cost_centers is None:
            csv_path = self.data_dir / 'cost_center_master.csv'
            self._cost_centers = pd.read_csv(csv_path)
        return self._cost_centers.copy()
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load ML training data.
        Contains labeled examples for classifier training.
        
        Returns:
            DataFrame with training data (6 sample rows).
        """
        if self._training_data is None:
            csv_path = self.data_dir / 'training_data.csv'
            self._training_data = pd.read_csv(csv_path)
        return self._training_data.copy()
    
    def get_erp_detail(self, entry_id: str) -> Dict:
        """
        Get detailed ERP record for drill-down analysis.
        Simulates fetching raw ERP system data.
        
        Args:
            entry_id: The journal entry ID to look up.
            
        Returns:
            Dictionary with ERP system details, or empty dict if not found.
        """
        # Mock ERP detail data for specific entries
        erp_details = {
            'JE001': {
                'erp_system': 'SAP ECC 6.0',
                'document_number': 'SAP-2024-0015230',
                'company_code': 'TN01',
                'fiscal_year': '2024',
                'fiscal_period': '10',
                'account': '500020',
                'cost_center': 'CC1001',
                'vendor': 'Grainger Industrial Supply',
                'vendor_id': 'V-100234',
                'amount': 45230.50,
                'currency': 'USD',
                'posting_date': '2024-10-15',
                'document_date': '2024-10-14',
                'entry_date': '2024-10-15',
                'reference': 'PO-2024-8834',
                'text': 'Direct Materials - Steel Components Q4 Procurement',
                'created_by': 'JSMITH',
                'approval_status': 'Posted',
                'workflow_id': 'WF-SAP-20241015-001'
            },
            'JE002': {
                'erp_system': 'JD Edwards EnterpriseOne',
                'document_number': 'JDE-RY-2024-00892',
                'company_code': 'RY01',
                'fiscal_year': '2024',
                'fiscal_period': '10',
                'account': '50010',
                'cost_center': 'CC2001',
                'vendor': 'Fastenal Company',
                'vendor_id': 'FAST001',
                'amount': 12875.00,
                'currency': 'USD',
                'posting_date': '2024-10-18',
                'document_date': '2024-10-17',
                'entry_date': '2024-10-18',
                'reference': 'REQ-2024-1122',
                'text': 'Raw Materials - Fasteners and Hardware',
                'created_by': 'MWILLIAMS',
                'approval_status': 'Posted',
                'workflow_id': 'WF-JDE-20241018-003'
            },
            'JE008': {
                'erp_system': 'JD Edwards EnterpriseOne',
                'document_number': 'JDE-RY-2024-01205',
                'company_code': 'RY01',
                'fiscal_year': '2024',
                'fiscal_period': '11',
                'account': '62500',
                'cost_center': 'CC2001',
                'vendor': 'McKinsey & Company',
                'vendor_id': 'MCKIN001',
                'amount': 50000.00,
                'currency': 'USD',
                'posting_date': '2024-11-10',
                'document_date': '2024-11-08',
                'entry_date': '2024-11-10',
                'reference': 'SOW-2024-STRATEGY',
                'text': 'Strategic Consulting - Process Optimization',
                'created_by': 'CFO_OFFICE',
                'approval_status': 'Pending Review',
                'workflow_id': 'WF-JDE-20241110-HIGH',
                'risk_flags': ['High Value', 'Consulting Category', 'Executive Approval Required']
            },
            'JE003': {
                'erp_system': 'IBM iSeries AS/400',
                'document_number': 'ISERIES-TN02-24-3341',
                'company_code': 'TN02',
                'fiscal_year': '2024',
                'fiscal_period': '10',
                'account': '620010',
                'cost_center': 'CC1002',
                'vendor': 'TravelPerk Business Travel',
                'vendor_id': 'TRVL-001',
                'amount': 8450.00,
                'currency': 'USD',
                'posting_date': '2024-10-20',
                'document_date': '2024-10-19',
                'entry_date': '2024-10-20',
                'reference': 'EXP-2024-Q4-SALES',
                'text': 'T&E - Sales Team Quarterly Conference',
                'created_by': 'AGARCIA',
                'approval_status': 'Posted',
                'workflow_id': 'WF-ISERIES-20241020-002'
            }
        }
        
        return erp_details.get(entry_id, {})
    
    def get_coa_by_legacy_code(self, legacy_code: str, source_erp: str) -> Optional[Dict]:
        """
        Look up TMHNA account by legacy ERP account code.
        
        Args:
            legacy_code: The legacy account code from source ERP.
            source_erp: The source ERP system name.
            
        Returns:
            Dictionary with TMHNA account info, or None if not found.
        """
        coa_df = self.load_coa_mapping()
        
        # Determine which column to search based on ERP source
        if 'SAP' in source_erp.upper():
            search_col = 'source_account_tmh_sap'
        else:
            search_col = 'source_account_raymond_jde'
        
        # Find matching row
        match = coa_df[coa_df[search_col].astype(str) == str(legacy_code)]
        
        if not match.empty:
            row = match.iloc[0]
            return {
                'tmhna_account_code': row['tmhna_account_code'],
                'tmhna_account_name': row['tmhna_account_name'],
                'account_category': row['account_category'],
                'description': row['description']
            }
        return None
    
    def refresh_data(self):
        """Clear cached data to force reload on next access."""
        self._journal_entries = None
        self._coa_mapping = None
        self._cost_centers = None
        self._training_data = None

