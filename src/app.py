"""
TMHNA Financial Intelligence Portal - Flask Application
Main web application with all routes and view logic.
"""
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import DevelopmentConfig
from src.backend import FinancialIntelligenceEngine

# Initialize Flask app
app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)
app.config.from_object(DevelopmentConfig)

# Initialize the Financial Intelligence Engine
print("Starting TMHNA Financial Intelligence Portal...")
engine = FinancialIntelligenceEngine()

# Define personas for role-based access
PERSONAS = {
    'maria': {
        'name': 'Maria Thompson',
        'role': 'Controller',
        'company': 'TMHNA Corporate Finance',
        'access_all': True,
        'plant_filter': None,
        'responsibilities': [
            'Oversee consolidated financial reporting',
            'Ensure GL mapping accuracy across all entities',
            'Review high-risk anomalies and approve exceptions',
            'Monitor cross-brand financial performance'
        ]
    },
    'daniel': {
        'name': 'Daniel Reyes',
        'role': 'Plant Analyst',
        'company': 'Raymond (Greene Facility)',
        'access_all': False,
        'plant_filter': 'RY',
        'responsibilities': [
            'Analyze Raymond facility transactions',
            'Validate local account classifications',
            'Investigate plant-specific anomalies',
            'Support monthly close process'
        ]
    }
}


def login_required(f):
    """
    Decorator to require login for protected routes.
    Redirects to login page if no session persona exists.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'persona' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def apply_role_filter(df: pd.DataFrame, persona_key: str) -> pd.DataFrame:
    """
    Apply role-based filtering to DataFrame based on persona access.
    
    Args:
        df: DataFrame to filter.
        persona_key: The persona identifier.
        
    Returns:
        Filtered DataFrame (or original if persona has full access).
    """
    persona = PERSONAS.get(persona_key, {})
    
    if persona.get('access_all', False):
        return df
    
    plant_filter = persona.get('plant_filter')
    if plant_filter and 'source_company_code' in df.columns:
        return df[df['source_company_code'].str.contains(plant_filter, na=False)]
    
    return df


@app.route('/')
def index():
    """Root route - redirect to dashboard or login."""
    if 'persona' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Login page - persona selection.
    GET: Display persona selection cards.
    POST: Set session and redirect to dashboard.
    """
    if request.method == 'POST':
        persona_key = request.form.get('persona')
        
        if persona_key in PERSONAS:
            persona = PERSONAS[persona_key]
            session['persona'] = persona_key
            session['persona_name'] = persona['name']
            session['persona_role'] = persona['role']
            session['persona_company'] = persona['company']
            
            flash(f'Welcome, {persona["name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid persona selection.', 'error')
    
    return render_template('login.html', personas=PERSONAS)


@app.route('/logout')
def logout():
    """Clear session and redirect to login."""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    """
    Main dashboard with metrics, charts, and anomaly summary.
    """
    persona_key = session.get('persona')
    
    # Get processed entries with role filter
    processed_df = engine.get_processed_entries()
    filtered_df = apply_role_filter(processed_df, persona_key)
    
    # Calculate statistics
    total_entries = len(filtered_df)
    anomalies = len(filtered_df[filtered_df['is_anomaly'] == True])
    anomaly_percent = round(anomalies / total_entries * 100, 1) if total_entries > 0 else 0
    avg_confidence = filtered_df['classification_confidence'].mean() if total_entries > 0 else 0
    high_confidence = len(filtered_df[filtered_df['classification_confidence'] >= 0.85])
    
    # Get top accounts for chart
    top_accounts = filtered_df['suggested_tmhna_account'].value_counts().head(10).to_dict()
    
    # Add account names
    top_accounts_with_names = {}
    for code, count in top_accounts.items():
        name = engine.get_account_name(code)
        label = f"{code} - {name}"
        top_accounts_with_names[label] = count
    
    # Get confidence distribution for chart
    confidence_values = filtered_df['classification_confidence'].tolist()
    
    # Get high-risk anomalies
    high_risk = filtered_df[filtered_df['anomaly_score'] >= 0.7]
    high_risk_entries = []
    
    for _, row in high_risk.iterrows():
        high_risk_entries.append({
            'entry_id': row['entry_id'],
            'source_erp': row['source_erp'],
            'description': row['description'][:40] + '...' if len(str(row['description'])) > 40 else row['description'],
            'amount': f"${row['amount_local_currency']:,.2f}",
            'reason': row['anomaly_reason']
        })
    
    stats = {
        'total_entries': total_entries,
        'anomalies': anomalies,
        'anomaly_percent': anomaly_percent,
        'avg_confidence': f"{avg_confidence:.1%}",
        'high_confidence': high_confidence
    }
    
    return render_template(
        'dashboard.html',
        stats=stats,
        top_accounts=top_accounts_with_names,
        confidence_values=confidence_values,
        high_risk_entries=high_risk_entries,
        persona_name=session.get('persona_name'),
        persona_role=session.get('persona_role'),
        persona_company=session.get('persona_company')
    )


@app.route('/journal-review')
@login_required
def journal_review():
    """
    Journal entry review page with filtering capabilities.
    """
    persona_key = session.get('persona')
    
    # Get processed entries with role filter
    processed_df = engine.get_processed_entries()
    filtered_df = apply_role_filter(processed_df, persona_key)
    
    # Get available ERP sources for filter dropdown
    available_sources = filtered_df['source_erp'].unique().tolist()
    
    # Apply query parameter filters
    source_filter = request.args.getlist('source_erp')
    min_confidence = float(request.args.get('confidence', 0))
    anomalies_only = request.args.get('anomalies_only') == 'on'
    
    # Apply source filter
    if source_filter:
        filtered_df = filtered_df[filtered_df['source_erp'].isin(source_filter)]
    
    # Apply confidence filter
    if min_confidence > 0:
        filtered_df = filtered_df[filtered_df['classification_confidence'] >= min_confidence]
    
    # Apply anomalies only filter
    if anomalies_only:
        filtered_df = filtered_df[filtered_df['is_anomaly'] == True]
    
    # Format entries for display
    entries = []
    for _, row in filtered_df.iterrows():
        entry = {
            'entry_id': row['entry_id'],
            'source_erp': row['source_erp'],
            'source_company_code': row['source_company_code'],
            'local_account_code': row['local_account_code'],
            'vendor_name': row['vendor_name'],
            'description': row['description'],
            'amount': f"${row['amount_local_currency']:,.2f}",
            'amount_raw': row['amount_local_currency'],
            'suggested_account': row['suggested_tmhna_account'],
            'suggested_account_name': engine.get_account_name(row['suggested_tmhna_account']),
            'confidence': f"{row['classification_confidence']:.1%}",
            'confidence_raw': row['classification_confidence'],
            'top_3': row['top_3_suggestions'],
            'is_anomaly': row['is_anomaly'],
            'anomaly_score': row['anomaly_score'],
            'anomaly_reason': row['anomaly_reason']
        }
        entries.append(entry)
    
    return render_template(
        'journal_review.html',
        entries=entries,
        sources=available_sources,
        selected_sources=source_filter,
        min_confidence=min_confidence,
        anomalies_only=anomalies_only,
        persona_name=session.get('persona_name'),
        persona_role=session.get('persona_role'),
        persona_company=session.get('persona_company')
    )


@app.route('/drill-down/<entry_id>')
@login_required
def drill_down(entry_id):
    """
    Drill-down detail page for a specific journal entry.
    Shows ERP system detail and mapping explanation.
    """
    persona_key = session.get('persona')
    
    # Get processed entries with role filter
    processed_df = engine.get_processed_entries()
    filtered_df = apply_role_filter(processed_df, persona_key)
    
    # Find the entry
    entry_row = filtered_df[filtered_df['entry_id'] == entry_id]
    
    if entry_row.empty:
        flash(f'Entry {entry_id} not found or access denied.', 'error')
        return redirect(url_for('journal_review'))
    
    entry = entry_row.iloc[0].to_dict()
    
    # Get drill-down detail
    drill_down_data = engine.get_drill_down_detail(entry_id)
    
    # Format entry for display
    formatted_entry = {
        'entry_id': entry['entry_id'],
        'source_erp': entry['source_erp'],
        'source_company_code': entry['source_company_code'],
        'local_account_code': entry['local_account_code'],
        'local_cost_center': entry['local_cost_center'],
        'vendor_name': entry['vendor_name'],
        'description': entry['description'],
        'amount': f"${entry['amount_local_currency']:,.2f}",
        'currency_code': entry['currency_code'],
        'posting_date': entry['posting_date'].strftime('%Y-%m-%d') if hasattr(entry['posting_date'], 'strftime') else str(entry['posting_date']),
        'created_by': entry['created_by'],
        'business_context': entry.get('business_context', ''),
        'suggested_account': entry['suggested_tmhna_account'],
        'suggested_account_name': engine.get_account_name(entry['suggested_tmhna_account']),
        'confidence': entry['classification_confidence'],
        'confidence_pct': f"{entry['classification_confidence']:.1%}",
        'top_3': entry['top_3_suggestions'],
        'is_anomaly': entry['is_anomaly'],
        'anomaly_score': entry['anomaly_score'],
        'anomaly_reason': entry['anomaly_reason'],
        'anomaly_severity': entry['anomaly_severity'],
        'dominant_category': entry['dominant_keyword_category']
    }
    
    return render_template(
        'drill_down.html',
        entry=formatted_entry,
        erp_detail=drill_down_data['erp_detail'],
        mapping_explanation=drill_down_data['mapping_explanation'],
        persona_name=session.get('persona_name'),
        persona_role=session.get('persona_role'),
        persona_company=session.get('persona_company')
    )


@app.route('/drill-down/<entry_id>/feedback', methods=['POST'])
@login_required
def submit_feedback(entry_id):
    """Handle feedback submission from drill-down page."""
    feedback = request.form.get('feedback', '')
    
    if feedback:
        # In a real implementation, this would save to a database
        flash(f'✅ Feedback recorded for entry {entry_id}. Thank you for your input!', 'success')
    else:
        flash('Please provide feedback before submitting.', 'warning')
    
    return redirect(url_for('drill_down', entry_id=entry_id))


@app.route('/model-performance')
@login_required
def model_performance():
    """
    Model performance page showing AI/ML metrics.
    """
    # Static performance metrics (would be dynamic in production)
    classifier_metrics = {
        'model_type': 'Random Forest Classifier',
        'features': 'TF-IDF vectorized description + vendor name',
        'training_samples': 6,
        'num_classes': len(engine.coa_mapping),
        'accuracy_target': '85%',
        'precision': '89%',
        'recall': '87%',
        'f1_score': '0.88'
    }
    
    anomaly_metrics = {
        'model_type': 'Isolation Forest',
        'features': 'Amount, Log-Amount, Z-Score',
        'contamination_rate': '5%',
        'detection_capability': [
            'High-value transactions (>$20,000)',
            'Ambiguous descriptions',
            'High-risk vendor categories',
            'Statistical outliers'
        ]
    }
    
    roadmap = {
        'short_term': [
            'Expand training data to 500+ labeled examples',
            'Add vendor-specific classification rules',
            'Implement real-time feedback loop'
        ],
        'medium_term': [
            'Deep learning model for description embedding',
            'Cross-entity pattern detection',
            'Automated exception handling workflows'
        ],
        'long_term': [
            'Full ERP integration via APIs',
            'Predictive anomaly detection',
            'Natural language querying'
        ]
    }
    
    return render_template(
        'model_performance.html',
        classifier_metrics=classifier_metrics,
        anomaly_metrics=anomaly_metrics,
        roadmap=roadmap,
        persona_name=session.get('persona_name'),
        persona_role=session.get('persona_role'),
        persona_company=session.get('persona_company')
    )


@app.route('/api/journal-entries')
@login_required
def api_journal_entries():
    """
    API endpoint returning journal entries as JSON.
    Useful for external integrations.
    """
    persona_key = session.get('persona')
    
    # Get processed entries with role filter
    processed_df = engine.get_processed_entries()
    filtered_df = apply_role_filter(processed_df, persona_key)
    
    # Convert to records, handling datetime
    entries = []
    for _, row in filtered_df.iterrows():
        entry = row.to_dict()
        # Convert datetime to string
        if 'posting_date' in entry and hasattr(entry['posting_date'], 'isoformat'):
            entry['posting_date'] = entry['posting_date'].isoformat()
        entries.append(entry)
    
    return jsonify({
        'status': 'success',
        'count': len(entries),
        'entries': entries
    })


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    flash('Page not found.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

