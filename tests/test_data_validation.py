import pytest
import pandas as pd
from src.data_validation import DataValidation


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "title": ["LG Refrigerator", "Samsung Washer"],
        "description": ["Double door cooling", "Front load washer"],
        "features": [["energy star", "ice maker"], ["steam wash"]],
        "categories": [["Appliances", "Refrigerators"], ["Appliances", "Washers"]],
        "parent_asin": ["B00111", "B00222"],
        "average_rating": [4.5, 4.2],
        "rating_number": [120, 85],
    })


def test_validation_passes_on_valid_data(valid_df):
    validator = DataValidation()
    report = validator.validate(valid_df)
    assert report["passed"] is True
    assert report["total_rows"] == 2
    assert report["duplicate_asins"] == 0


def test_validation_fails_on_missing_columns(valid_df):
    validator = DataValidation()
    invalid_df = valid_df.drop(columns=["title"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validator.validate(invalid_df)


def test_validation_fails_on_invalid_ratings(valid_df):
    validator = DataValidation()
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "average_rating"] = 6.5
    with pytest.raises(ValueError, match="Ratings outside valid range"):
        validator.validate(invalid_df)


def test_validation_fails_on_negative_counts(valid_df):
    validator = DataValidation()
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "rating_number"] = -5
    with pytest.raises(ValueError, match="Found negative rating counts"):
        validator.validate(invalid_df)
