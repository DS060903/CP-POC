#!/usr/bin/env python
"""
Generate Mock Data Script
Expands the mock dataset with additional journal entries for testing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def generate_journal_entries(num_entries=100):
    """
    Generate expanded mock journal entries.
    
    Args:
        num_entries: Number of entries to generate.
        
    Returns:
        DataFrame with generated entries.
    """
    # Configuration
    erp_systems = ['SAP_ECC', 'JDE', 'ISERIES']
    company_codes = {
        'SAP_ECC': ['TN01', 'TN02'],
        'JDE': ['RY01', 'RY02'],
        'ISERIES': ['TN01', 'TN02', 'RY01']
    }
    
    vendors = {
        'materials': [
            'Grainger Industrial Supply', 'Fastenal Company', 'McMaster-Carr',
            'MSC Industrial Direct', 'Caterpillar Parts Inc', 'Parker Hannifin'
        ],
        'labor': [
            'ADP Payroll Services', 'Paychex Inc', 'Ceridian HCM'
        ],
        'utilities': [
            'Duke Energy Corporation', 'Indiana Michigan Power', 'National Grid'
        ],
        'travel': [
            'TravelPerk Business Travel', 'Concur Technologies', 'American Express Travel'
        ],
        'consulting': [
            'McKinsey & Company', 'Deloitte Consulting', 'Boston Consulting Group',
            'Accenture', 'PwC Advisory'
        ],
        'freight': [
            'FedEx Freight Services', 'UPS Supply Chain', 'XPO Logistics',
            'C.H. Robinson'
        ],
        'maintenance': [
            'Grainger Industrial Supply', 'Fastenal Company', 'Applied Industrial'
        ]
    }
    
    descriptions = {
        'materials': [
            'Direct Materials - Steel Components',
            'Raw Materials - Fasteners and Hardware',
            'Production Materials - Assembly Components',
            'Direct Materials - Forklift Parts',
            'Raw Materials - Hydraulic Components'
        ],
        'labor': [
            'Direct Labor - Assembly Line Workers',
            'Direct Labor - Production Staff',
            'Labor Costs - Manufacturing Team',
            'Payroll - Plant Workers'
        ],
        'utilities': [
            'Utilities - Plant Electricity',
            'Utilities - Natural Gas',
            'Plant Utilities - Monthly',
            'Facility Energy Costs'
        ],
        'travel': [
            'T&E - Sales Team Conference',
            'Travel Expenses - Customer Visit',
            'Business Travel - Trade Show',
            'Travel - Regional Meeting'
        ],
        'consulting': [
            'Strategic Consulting - Process Optimization',
            'Advisory Services - Operations Review',
            'Consulting - IT Implementation',
            'Professional Services - Strategy'
        ],
        'freight': [
            'Freight - Inbound Materials Shipping',
            'Shipping - Outbound Products',
            'Logistics - Supply Chain',
            'Freight Costs - Delivery'
        ],
        'maintenance': [
            'Maintenance - Equipment Repair Parts',
            'Repairs - Conveyor System',
            'Equipment Maintenance - Quarterly',
            'Facility Repairs - General'
        ],
        'ambiguous': [
            'Misk Exp - Oct',
            'Misc Charges',
            'Adjustment - Q4',
            'Other Expenses',
            'Sundry Items'
        ]
    }
    
    account_codes = {
        'materials': {'SAP_ECC': '500020', 'JDE': '50010', 'ISERIES': '500020'},
        'labor': {'SAP_ECC': '510030', 'JDE': '51010', 'ISERIES': '510030'},
        'utilities': {'SAP_ECC': '64020', 'JDE': '64020', 'ISERIES': '64020'},
        'travel': {'SAP_ECC': '620010', 'JDE': '62010', 'ISERIES': '620010'},
        'consulting': {'SAP_ECC': '62500', 'JDE': '62500', 'ISERIES': '62500'},
        'freight': {'SAP_ECC': '52010', 'JDE': '52010', 'ISERIES': '52010'},
        'maintenance': {'SAP_ECC': '63010', 'JDE': '63010', 'ISERIES': '63010'},
        'ambiguous': {'SAP_ECC': '64010', 'JDE': '64010', 'ISERIES': '64010'}
    }
    
    cost_centers = ['CC1001', 'CC1002', 'CC1003', 'CC2001', 'CC2002', 'CC2003', 'CC3001', 'CC3002']
    users = ['jsmith', 'mwilliams', 'agarcia', 'jdoe', 'hrpayroll', 'facilitymgr', 'logistics', 'maintenance', 'cfo_office', 'salesteam']
    
    entries = []
    base_date = datetime(2024, 10, 1)
    
    for i in range(num_entries):
        entry_id = f'JE{str(i + 1).zfill(3)}'
        
        # Select category (weighted for realistic distribution)
        category = random.choices(
            ['materials', 'labor', 'utilities', 'travel', 'consulting', 'freight', 'maintenance', 'ambiguous'],
            weights=[25, 15, 10, 15, 5, 15, 10, 5]
        )[0]
        
        # Select ERP and company code
        erp = random.choice(erp_systems)
        company_code = random.choice(company_codes[erp])
        
        # Generate amount based on category
        if category == 'labor':
            amount = round(random.uniform(50000, 150000), 2)
        elif category == 'consulting':
            amount = round(random.uniform(10000, 75000), 2)
        elif category == 'utilities':
            amount = round(random.uniform(5000, 25000), 2)
        elif category == 'materials':
            amount = round(random.uniform(5000, 80000), 2)
        else:
            amount = round(random.uniform(500, 15000), 2)
        
        # Random date within 3 months
        posting_date = base_date + timedelta(days=random.randint(0, 90))
        
        # Build entry
        entry = {
            'entry_id': entry_id,
            'source_erp': erp,
            'source_company_code': company_code,
            'local_account_code': account_codes[category][erp],
            'local_cost_center': random.choice(cost_centers),
            'vendor_name': random.choice(vendors.get(category, vendors['materials'])),
            'description': random.choice(descriptions[category]),
            'amount_local_currency': amount,
            'currency_code': 'USD',
            'posting_date': posting_date.strftime('%Y-%m-%d'),
            'created_by': random.choice(users),
            'business_context': f'{category.title()} transaction for {company_code}'
        }
        
        entries.append(entry)
    
    return pd.DataFrame(entries)


def main():
    """Generate and save expanded mock data."""
    print("Generating expanded mock data...")
    
    # Generate entries
    df = generate_journal_entries(100)
    
    # Save to CSV
    output_path = DATA_DIR / 'journal_entries_expanded.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} journal entries")
    print(f"Saved to: {output_path}")
    
    # Print sample
    print("\nSample entries:")
    print(df.head(5).to_string())


if __name__ == '__main__':
    main()

