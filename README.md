# TMHNA Financial Intelligence Portal

A production-ready Flask web application demonstrating TMHNA's unified financial intelligence architecture.

## Features

- **Multi-ERP Data Integration**: Ingests mock financial data from SAP, JDE, and iSeries
- **ML-Powered Classification**: Automatically classifies journal entries to unified Chart of Accounts
- **Anomaly Detection**: Flags suspicious transactions using Isolation Forest
- **Role-Based Access**: Controller vs Plant Analyst views with data filtering
- **Interactive Dashboards**: Plotly charts, metrics, and drill-down analysis

## Quick Start

### 1. Create a Virtual Environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Install Dependencies & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py

# Open in browser
http://localhost:5000
```

> **Note:** To deactivate the virtual environment when done, simply run `deactivate`

## Project Structure

```
tmhna_financial_poc/
├── config/
│   └── config.py              # Flask configuration
├── data/                      # CSV mock datasets
│   ├── journal_entries_raw.csv
│   ├── chart_of_accounts.csv
│   ├── cost_center_master.csv
│   └── training_data.csv
├── models/                    # Trained ML models (auto-generated)
├── notebooks/
│   ├── generate_mock_data.py  # Data generation script
│   └── train_models.py        # Model training script
├── src/
│   ├── app.py                 # Flask application
│   ├── backend.py             # Intelligence engine
│   ├── data_loader.py         # Data loading
│   ├── classifier.py          # ML classifier
│   ├── anomaly_detector.py    # Anomaly detection
│   └── nlp_processor.py       # NLP processing
├── static/
│   ├── css/style.css
│   └── js/main.js
├── templates/                 # Jinja2 templates
├── requirements.txt
├── run.py                     # Entry point
└── README.md
```

## Personas

### Maria Thompson - Controller
- Full access to all company data
- Oversees consolidated financial reporting
- Reviews high-risk anomalies

### Daniel Reyes - Plant Analyst
- Filtered access to Raymond facility (RY*) data only
- Analyzes plant-specific transactions
- Supports monthly close process

## Pages

1. **Dashboard**: Key metrics, charts, high-risk anomalies
2. **Journal Review**: Filter and review classified entries
3. **Drill-Down**: Detailed ERP record analysis
4. **Model Performance**: AI/ML metrics and roadmap

## Technology Stack

- **Backend**: Flask 2.3, Python 3.9+
- **ML**: scikit-learn (Random Forest, Isolation Forest)
- **Frontend**: Jinja2, Plotly.js, Custom CSS
- **Data**: pandas, CSV files

## Demo Walkthrough (60 seconds)

1. Login as Maria (Controller) - see all data
2. View Dashboard metrics and anomaly table
3. Click on JE008 (high-risk consulting entry)
4. Review ERP detail and mapping explanation
5. Logout and login as Daniel (Plant Analyst)
6. Notice filtered view (only RY01/RY02 entries)
7. Visit Model Performance page

## Configuration

Key settings in `config/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MIN_CONFIDENCE_AUTO_APPROVE` | 0.95 | Auto-approve threshold |
| `MIN_CONFIDENCE_FLAG_REVIEW` | 0.70 | Manual review threshold |
| `ANOMALY_THRESHOLD` | 0.70 | Anomaly flagging threshold |

## Development

```bash
# Train models manually
python notebooks/train_models.py

# Generate more mock data
python notebooks/generate_mock_data.py
```

## License

© 2024 Toyota Material Handling North America. Internal use only.

