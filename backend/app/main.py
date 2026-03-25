import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from app.routes import analysis, aws_routes, health, seed_routes

app = FastAPI(
    title="Sentiment Analysis API",
    description="Multi-level sentiment analysis using BERT-BiLSTM-Attention",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analysis.router)
app.include_router(aws_routes.router)
app.include_router(seed_routes.router)

frontend_build = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(frontend_build):
    app.mount("/", StaticFiles(directory=frontend_build, html=True), name="frontend")


@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "document_analysis": "/api/analysis/document",
            "sentence_analysis": "/api/analysis/sentence",
            "aspect_analysis": "/api/analysis/aspect",
            "full_analysis": "/api/analysis/full",
            "batch_analysis": "/api/analysis/batch",
            "aws_status": "/api/aws/status",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
