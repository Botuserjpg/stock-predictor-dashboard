"""
API Keys Configuration for Stock Predictor Pro
Keys are read from Streamlit secrets (st.secrets) when available, otherwise
from environment variables or the local .env file. Never hardcode real keys here.
"""

import os
from typing import Dict, Any, Optional

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:  # pragma: no cover - streamlit may not be installed (Flask-only)
    _HAS_STREAMLIT = False


def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a secret from st.secrets first, then environment variables."""
    if _HAS_STREAMLIT:
        try:
            value = st.secrets.get(name)
            if value:
                return str(value)
        except Exception:
            pass
    return os.getenv(name, default)


class APIConfig:
    """API Keys Configuration"""

    # News Data APIs
    NEWSAPI_KEY = _get_secret('NEWSAPI_KEY')
    ALPHAVANTAGE_KEY = _get_secret('ALPHAVANTAGE_KEY')
    FINNHUB_KEY = _get_secret('FINNHUB_KEY')
    GNEWS_KEY = _get_secret('GNEWS_KEY')

    # Financial Data APIs
    FMP_KEY = _get_secret('FMP_KEY')
    POLYGON_KEY = _get_secret('POLYGON_KEY')
    TIINGO_KEY = _get_secret('TIINGO_KEY')

    # Sentiment & Social Media
    TWITTER_BEARER_TOKEN = _get_secret('TWITTER_BEARER_TOKEN')
    REDDIT_CLIENT_ID = _get_secret('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = _get_secret('REDDIT_CLIENT_SECRET')
    REDDIT_USERNAME = _get_secret('REDDIT_USERNAME')
    REDDIT_PASSWORD = _get_secret('REDDIT_PASSWORD')

    # Alternative Data
    RAVENPACK_KEY = _get_secret('RAVENPACK_KEY', 'YOUR_RAVENPACK_KEY_HERE')
    ACCERN_KEY = _get_secret('ACCERN_KEY', 'YOUR_ACCERN_KEY_HERE')
    
    @classmethod
    def get_all_keys(cls) -> Dict[str, Any]:
        """Get all API keys as dictionary"""
        return {
            'newsapi': cls.NEWSAPI_KEY,
            'alphavantage': cls.ALPHAVANTAGE_KEY,
            'finnhub': cls.FINNHUB_KEY,
            'gnews': cls.GNEWS_KEY,
            'fmp': cls.FMP_KEY,
            'polygon': cls.POLYGON_KEY,
            'tiingo': cls.TIINGO_KEY,
            'twitter': cls.TWITTER_BEARER_TOKEN,
            'reddit': {
                'client_id': cls.REDDIT_CLIENT_ID,
                'client_secret': cls.REDDIT_CLIENT_SECRET,
                'username': cls.REDDIT_USERNAME,
                'password': cls.REDDIT_PASSWORD
            },
            'ravenpack': cls.RAVENPACK_KEY,
            'accern': cls.ACCERN_KEY
        }
    
    @classmethod
    def validate_keys(cls) -> Dict[str, bool]:
        """Validate which API keys are available"""
        available = {}
        keys = cls.get_all_keys()
        
        for service, key in keys.items():
            if service == 'reddit':
                available[service] = bool(key['client_id']) and bool(key['client_secret'])
            else:
                available[service] = bool(key) and isinstance(key, str) and not key.startswith('YOUR_')
        
        return available
