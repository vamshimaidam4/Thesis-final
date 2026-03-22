import json
import os

SEED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "seed_data", "sample_reviews.json")


def load_seed_reviews() -> list:
    """Load seed review data for demo purposes."""
    if os.path.exists(SEED_DATA_PATH):
        with open(SEED_DATA_PATH, "r") as f:
            return json.load(f)
    return []


def get_seed_review(review_id: int) -> dict:
    """Get a specific seed review by ID."""
    reviews = load_seed_reviews()
    for review in reviews:
        if review["id"] == review_id:
            return review
    return None


def get_seed_stats() -> dict:
    """Get statistics about seed data."""
    reviews = load_seed_reviews()
    if not reviews:
        return {"count": 0}

    sentiments = {}
    categories = {}
    for r in reviews:
        s = r.get("expected_sentiment", "Unknown")
        sentiments[s] = sentiments.get(s, 0) + 1
        c = r.get("category", "Unknown")
        categories[c] = categories.get(c, 0) + 1

    return {
        "count": len(reviews),
        "sentiments": sentiments,
        "categories": categories,
        "avg_length": sum(len(r["text"].split()) for r in reviews) / len(reviews),
    }
