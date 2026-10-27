import pytest
from src.evaluate import Evaluator


def test_evaluator_metrics():
    evaluator = Evaluator()
    sample_recommendations = [
        {
            "parent_asin": "B01",
            "title": "Ice Maker A",
            "average_rating": 4.5,
            "categories": ["Appliances", "Ice Makers"],
            "similarity_score": 0.92,
        },
        {
            "parent_asin": "B02",
            "title": "Ice Maker B",
            "average_rating": 4.1,
            "categories": ["Appliances", "Ice Makers"],
            "similarity_score": 0.88,
        },
    ]
    metrics = evaluator.evaluate(sample_recommendations)
    assert metrics["recommendation_count"] == 2
    assert metrics["average_rating"] == 4.3
    assert metrics["mean_similarity"] == 0.90
    assert metrics["unique_categories_covered"] == 2


def test_evaluator_empty():
    evaluator = Evaluator()
    metrics = evaluator.evaluate([])
    assert metrics["recommendation_count"] == 0
