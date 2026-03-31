import pytest
from app.services.seed_service import load_seed_reviews, get_seed_review, get_seed_stats


def test_load_seed_reviews():
    reviews = load_seed_reviews()
    assert len(reviews) == 15
    assert all("text" in r for r in reviews)
    assert all("id" in r for r in reviews)
    assert all("star_rating" in r for r in reviews)
    assert all("expected_sentiment" in r for r in reviews)


def test_seed_review_ids_unique():
    reviews = load_seed_reviews()
    ids = [r["id"] for r in reviews]
    assert len(ids) == len(set(ids))


def test_get_seed_review_exists():
    review = get_seed_review(1)
    assert review is not None
    assert review["id"] == 1
    assert len(review["text"]) > 0


def test_get_seed_review_not_found():
    review = get_seed_review(9999)
    assert review is None


def test_seed_stats():
    stats = get_seed_stats()
    assert stats["count"] == 15
    assert "sentiments" in stats
    assert "Positive" in stats["sentiments"]
    assert "Negative" in stats["sentiments"]
    assert "Neutral" in stats["sentiments"]
    assert stats["avg_length"] > 0


def test_seed_reviews_have_aspects():
    reviews = load_seed_reviews()
    for review in reviews:
        assert "expected_aspects" in review
        assert len(review["expected_aspects"]) > 0


def test_seed_reviews_star_ratings_valid():
    reviews = load_seed_reviews()
    for review in reviews:
        assert 1 <= review["star_rating"] <= 5


def test_seed_reviews_sentiments_valid():
    reviews = load_seed_reviews()
    valid = {"Positive", "Negative", "Neutral"}
    for review in reviews:
        assert review["expected_sentiment"] in valid
