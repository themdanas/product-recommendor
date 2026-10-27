import pytest
import pandas as pd
from src.feature_engineering import FeatureEngineering


def test_clean_text_removes_punctuation_and_lowercases():
    fe = FeatureEngineering()
    raw = "LG 4-Door Stainless Refrigerator (Brand New!) -- 100% Reliable."
    cleaned = fe.clean_text(raw)
    assert cleaned == "lg 4 door stainless refrigerator brand new 100 reliable"


def test_clean_text_handles_none():
    fe = FeatureEngineering()
    assert fe.clean_text(None) == ""


def test_stringify_field_handles_lists_and_nans():
    fe = FeatureEngineering()
    assert fe._stringify_field(["Appliances", "Washers"]) == "Appliances Washers"
    assert fe._stringify_field("nan") == ""
    assert fe._stringify_field(None) == ""


def test_create_combined_text_filters_low_ratings():
    fe = FeatureEngineering()
    df = pd.DataFrame({
        "title": ["Product A", "Product B"],
        "features": [["feat1"], ["feat2"]],
        "categories": [["cat1"], ["cat2"]],
        "description": ["desc1", "desc2"],
        "rating_number": [10, 2],
    })
    processed = fe.create_combined_text(df, min_ratings=5)
    assert len(processed) == 1
    assert processed.iloc[0]["title"] == "Product A"
    assert "feat1" in processed.iloc[0]["combined_text"]
