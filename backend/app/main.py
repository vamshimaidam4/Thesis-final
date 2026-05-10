import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from app.routes import analysis, aws_routes, health, sagemaker_routes, seed_routes, training  # noqa: E402

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
app.include_router(training.router)
app.include_router(sagemaker_routes.router)

frontend_build = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(frontend_build):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(frontend_build, "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        file_path = os.path.join(frontend_build, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_build, "index.html"))
else:
    @app.get("/")
    def root():
        return {
            "message": "Sentiment Analysis API",
            "docs": "/docs",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
