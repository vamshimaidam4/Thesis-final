from fastapi import APIRouter, HTTPException
from app.services.seed_service import load_seed_reviews, get_seed_review, get_seed_stats

router = APIRouter(prefix="/api/seed", tags=["seed"])


@router.get("/reviews")
def list_seed_reviews():
    """List all seed reviews."""
    reviews = load_seed_reviews()
    return {"reviews": reviews, "count": len(reviews)}


@router.get("/reviews/{review_id}")
def get_review(review_id: int):
    """Get a specific seed review."""
    review = get_seed_review(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    return review


@router.get("/stats")
def seed_stats():
    """Get seed data statistics."""
    return get_seed_stats()
