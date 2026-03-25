from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.aws_service import (
    check_aws_connection,
    ensure_bucket_exists,
    upload_model_to_s3,
    download_model_from_s3,
    list_models_in_s3,
)

router = APIRouter(prefix="/api/aws", tags=["aws"])


class UploadRequest(BaseModel):
    local_path: str
    s3_key: str


class DownloadRequest(BaseModel):
    s3_key: str
    local_path: str


@router.get("/status")
def aws_status():
    """Check AWS connection status."""
    return check_aws_connection()


@router.post("/bucket/create")
def create_bucket():
    """Ensure the S3 bucket exists."""
    success = ensure_bucket_exists()
    if success:
        return {"message": "Bucket ready"}
    raise HTTPException(status_code=500, detail="Failed to create bucket")


@router.post("/models/upload")
def upload_model(req: UploadRequest):
    """Upload a model checkpoint to S3."""
    success = upload_model_to_s3(req.local_path, req.s3_key)
    if success:
        return {"message": f"Uploaded to {req.s3_key}"}
    raise HTTPException(status_code=500, detail="Upload failed")


@router.post("/models/download")
def download_model(req: DownloadRequest):
    """Download a model checkpoint from S3."""
    success = download_model_from_s3(req.s3_key, req.local_path)
    if success:
        return {"message": f"Downloaded to {req.local_path}"}
    raise HTTPException(status_code=500, detail="Download failed")


@router.get("/models/list")
def list_models():
    """List all models in S3."""
    models = list_models_in_s3()
    return {"models": models}
