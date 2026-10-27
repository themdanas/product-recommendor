import pytest
from src.recommender import Recommender


@pytest.fixture(scope="module")
def recommender():
    return Recommender()


def test_recommender_initialization(recommender):
    assert len(recommender.df) > 0
    assert recommender.model is not None
    assert recommender.vectorizer is not None


def test_recommend_by_title(recommender):
    results = recommender.recommend("Ice Maker", top_k=5)
    assert len(results) > 0
    assert len(results) <= 5
    for item in results:
        assert "title" in item
        assert "parent_asin" in item
        assert "similarity_score" in item
        assert item["similarity_score"] <= 1.0


def test_recommend_by_query_semantic(recommender):
    results = recommender.recommend_by_query("refrigerator water filter replacement", top_k=4)
    assert len(results) == 4
    for item in results:
        assert "title" in item
        assert "image_url" in item
        assert item["image_url"].startswith("http")


def test_search_products(recommender):
    matches = recommender.search_products("Dryer", limit=5)
    assert len(matches) > 0
    assert len(matches) <= 5
    assert "dryer" in matches[0]["title"].lower()


def test_empty_query_returns_empty_list(recommender):
    assert recommender.recommend("") == []
    assert recommender.recommend_by_query("   ") == []
