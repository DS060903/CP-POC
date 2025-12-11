"""
Flask Configuration Classes
Defines development, production, and testing configurations.
"""
import os
from pathlib import Path


class Config:
    """Base configuration with default settings."""
    
    # Base directory of the application
    BASE_DIR = Path(__file__).parent.parent
    
    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY', 'tmhna-financial-dev-key-change-in-production')
    
    # Data and model directories
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    
    # ML Configuration thresholds
    MIN_CONFIDENCE_AUTO_APPROVE = 0.95
    MIN_CONFIDENCE_FLAG_REVIEW = 0.7
    ANOMALY_THRESHOLD = 0.7
    
    # Session configuration
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    
    # Database (not used in CSV-based implementation but included for completeness)
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    """Development configuration with debug enabled."""
    
    DEBUG = True
    TESTING = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///dev.db'
    
    # More verbose logging in development
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration with security hardening."""
    
    DEBUG = False
    TESTING = False
    
    # In production, SECRET_KEY must be set via environment variable
    SECRET_KEY = os.environ.get('SECRET_KEY', 'tmhna-prod-change-this-in-production')
    
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///prod.db')
    
    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    LOG_LEVEL = 'WARNING'


class TestingConfig(Config):
    """Testing configuration with in-memory database."""
    
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False
    
    LOG_LEVEL = 'DEBUG'


# Configuration dictionary for easy access
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

