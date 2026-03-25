import torch
from fastapi import APIRouter
from app.config import config

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health_check():
    """Health check endpoint."""
    device_info = {"device": config.device}
    if torch.cuda.is_available():
        device_info["gpu"] = torch.cuda.get_device_name(0)
        device_info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1e9, 1
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_info["gpu"] = "Apple Silicon (MPS)"

    return {
        "status": "healthy",
        "model": config.model.bert_model_name,
        "device": device_info,
    }
