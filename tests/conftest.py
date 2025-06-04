import pytest
import pandas as pd
import os
import sys

# Add the package root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def sample_data():
    """Create a sample dataset for testing"""
    return pd.DataFrame({
        'ISO3': ['USA', 'CHN', 'DEU'] * 2,
        'countryname': ['United States', 'China', 'Germany'] * 2,
        'year': [2020, 2020, 2020, 2021, 2021, 2021],
        'rGDP': [1000, 800, 600, 1050, 850, 620],
        'infl': [2.0, 3.0, 1.5, 2.5, 3.5, 1.8]
    }) 