#!/usr/bin/env python
"""
TMHNA Financial Intelligence Portal - Entry Point
Run this file to start the application: python run.py
"""
import sys
from pathlib import Path

# Ensure the project root is in the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        ('flask', 'Flask'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('numpy', 'numpy'),
    ]
    
    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("=" * 60)
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install them with:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 60)
        return False
    return True


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        project_root / 'data',
        project_root / 'models',
        project_root / 'static' / 'css',
        project_root / 'static' / 'js',
        project_root / 'static' / 'img',
        project_root / 'templates',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def check_data_files():
    """Check if data files exist."""
    data_dir = project_root / 'data'
    required_files = [
        'journal_entries_raw.csv',
        'chart_of_accounts.csv',
        'cost_center_master.csv',
        'training_data.csv'
    ]
    
    missing = []
    for filename in required_files:
        if not (data_dir / filename).exists():
            missing.append(filename)
    
    if missing:
        print("Warning: Some data files are missing:")
        for f in missing:
            print(f"  - {f}")
        print("The application may not function correctly.")
        return False
    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("  TMHNA Financial Intelligence Portal")
    print("  Production-Ready Flask Application")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("[1/4] Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("      ✓ All dependencies installed")
    
    # Ensure directories
    print("[2/4] Ensuring directories exist...")
    ensure_directories()
    print("      ✓ Directories ready")
    
    # Check data files
    print("[3/4] Checking data files...")
    check_data_files()
    print("      ✓ Data files present")
    
    # Start application
    print("[4/4] Starting Flask application...")
    print()
    print("-" * 60)
    print("  Server starting at: http://localhost:5000")
    print("  Press Ctrl+C to stop the server")
    print("-" * 60)
    print()
    
    # Import and run the app
    from src.app import app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )


if __name__ == '__main__':
    main()

