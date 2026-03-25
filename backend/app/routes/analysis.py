from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.inference_service import (
    analyze_document_sentiment,
    analyze_sentence_sentiment,
    analyze_aspects,
    analyze_full,
    analyze_with_bilstm,
)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Review text to analyze")
    model: Optional[str] = Field("hmgs", description="Model to use: hmgs or bilstm")


class BatchReviewRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=50)


@router.post("/document")
def document_sentiment(req: ReviewRequest):
    """Analyze document-level sentiment."""
    try:
        return analyze_document_sentiment(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentence")
def sentence_sentiment(req: ReviewRequest):
    """Analyze sentence-level sentiment."""
    try:
        return analyze_sentence_sentiment(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/aspect")
def aspect_sentiment(req: ReviewRequest):
    """Extract and analyze aspect-level sentiment."""
    try:
        return analyze_aspects(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full")
def full_analysis(req: ReviewRequest):
    """Run all three levels of sentiment analysis."""
    try:
        if req.model == "bilstm":
            bilstm_result = analyze_with_bilstm(req.text)
            return {"text": req.text, "bilstm_analysis": bilstm_result}
        return analyze_full(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
def batch_analysis(req: BatchReviewRequest):
    """Analyze multiple reviews at once."""
    try:
        results = []
        for text in req.reviews:
            result = analyze_full(text)
            results.append(result)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
