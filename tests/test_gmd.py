import pytest
import pandas as pd
from global_macro_data import (
    gmd,
    get_available_versions,
    get_current_version,
    list_variables,
    list_countries,
    VALID_VARIABLES
)

def test_get_available_versions():
    """Test getting available versions"""
    versions = get_available_versions()
    assert isinstance(versions, list)
    assert len(versions) > 0
    assert all(isinstance(v, str) for v in versions)
    assert all(len(v.split('_')) == 2 for v in versions)

def test_get_current_version():
    """Test getting current version"""
    version = get_current_version()
    assert isinstance(version, str)
    assert len(version.split('_')) == 2

def test_list_variables(capsys):
    """Test listing variables"""
    list_variables()
    captured = capsys.readouterr()
    assert "Available variables" in captured.out
    for var in VALID_VARIABLES:
        assert var in captured.out

def test_list_countries(capsys):
    """Test listing countries"""
    list_countries()
    captured = capsys.readouterr()
    assert "Country and territories" in captured.out
    assert "Code" in captured.out

def test_gmd_default():
    """Test default gmd call"""
    df = gmd()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert all(col in df.columns for col in ["ISO3", "countryname", "year"])

def test_gmd_version():
    """Test gmd with specific version"""
    version = get_current_version()
    df = gmd(version=version)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_gmd_country():
    """Test gmd with specific country"""
    df = gmd(country="USA")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert all(df["ISO3"] == "USA")

def test_gmd_countries():
    """Test gmd with multiple countries"""
    df = gmd(country=["USA", "CHN"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert set(df["ISO3"].unique()) == {"USA", "CHN"}

def test_gmd_variables():
    """Test gmd with specific variables"""
    df = gmd(variables=["rGDP", "infl"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert all(col in df.columns for col in ["rGDP", "infl"])

def test_gmd_raw():
    """Test gmd with raw data option"""
    df = gmd(variables="rGDP", raw=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "rGDP" in df.columns

def test_gmd_combinations():
    """Test gmd with multiple parameters"""
    df = gmd(
        version=get_current_version(),
        country=["USA", "CHN"],
        variables=["rGDP", "infl"]
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert set(df["ISO3"].unique()) == {"USA", "CHN"}
    assert all(col in df.columns for col in ["rGDP", "infl"])

def test_gmd_invalid_version():
    """Test gmd with invalid version"""
    with pytest.raises(ValueError):
        gmd(version="invalid_version")

def test_gmd_invalid_country():
    """Test gmd with invalid country"""
    with pytest.raises(ValueError):
        gmd(country="INVALID")

def test_gmd_invalid_variable():
    """Test gmd with invalid variable"""
    with pytest.raises(ValueError):
        gmd(variables="INVALID")

def test_gmd_raw_multiple_variables():
    """Test gmd raw option with multiple variables"""
    with pytest.raises(ValueError):
        gmd(variables=["rGDP", "infl"], raw=True)

def test_gmd_raw_no_variable():
    """Test gmd raw option without variable"""
    with pytest.raises(ValueError):
        gmd(raw=True) 