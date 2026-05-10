"""SageMaker training endpoints exposed to the frontend.

The running ECS task uses its task role credentials (not env vars) to:
  - submit a SageMaker PyTorch training job
  - poll job status
  - list recent jobs in this account
  - repack the trained model.tar.gz into the inference/current/ S3 prefix
    so the inference backend picks up the new weights without a rebuild

Code packaging note: the SageMaker source dir lives at aws/sagemaker_train/
in the repo. The Docker image bundles the whole project, so the path is
resolved relative to this file's location at runtime.
"""
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import config

router = APIRouter(prefix="/api/sagemaker", tags=["sagemaker"])

SOURCE_DIR = (Path(__file__).resolve().parents[3] / "aws" / "sagemaker_train")
SAGEMAKER_ROLE_NAME = "SageMakerSentimentExecutionRole"
INFERENCE_PREFIX = "inference/current/"
DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"


class TrainingRequest(BaseModel):
    instance_type: str = Field(default=DEFAULT_INSTANCE_TYPE)
    model: str = Field(default="both", pattern="^(hmgs|bilstm|both)$")
    epochs: int = Field(default=3, ge=1, le=20)
    batch_size: int = Field(default=16, ge=1, le=64)
    train_samples: int = Field(default=10000, ge=300, le=200000)
    val_samples: int = Field(default=1000, ge=100, le=20000)
    max_len: int = Field(default=128, ge=32, le=512)
    hf_config: str = Field(default="raw_review_All_Beauty")
    max_run_seconds: int = Field(default=4 * 3600, ge=600, le=24 * 3600)


def _boto_session():
    return boto3.Session(region_name=config.aws.region)


def _aws_account_id() -> Optional[str]:
    try:
        return _boto_session().client("sts").get_caller_identity()["Account"]
    except (ClientError, Exception):
        return None


def _resolve_sagemaker_role_arn() -> str:
    account = _aws_account_id()
    if not account:
        raise HTTPException(500, "Cannot resolve AWS account id; check task role")
    arn = f"arn:aws:iam::{account}:role/{SAGEMAKER_ROLE_NAME}"
    iam = _boto_session().client("iam")
    try:
        iam.get_role(RoleName=SAGEMAKER_ROLE_NAME)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchEntity":
            raise HTTPException(
                503,
                f"SageMaker execution role {SAGEMAKER_ROLE_NAME} not found. "
                "It is created by the deploy workflow; redeploy or create it manually.",
            )
        raise HTTPException(500, f"IAM error: {e.response['Error']['Message']}")
    return arn


def _ensure_bucket():
    bucket = config.aws.s3_bucket
    region = config.aws.region
    s3 = _boto_session().client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
        return bucket
    except ClientError:
        kwargs = {"Bucket": bucket}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**kwargs)
        return bucket


@router.get("/status")
def sagemaker_status():
    """Quick health check: does the task role have the perms we need?"""
    info = {
        "region": config.aws.region,
        "bucket": config.aws.s3_bucket,
        "source_dir_exists": SOURCE_DIR.exists(),
        "source_dir": str(SOURCE_DIR),
    }
    sess = _boto_session()
    try:
        info["account_id"] = sess.client("sts").get_caller_identity()["Account"]
    except Exception as e:
        info["account_id"] = None
        info["sts_error"] = str(e)
    try:
        sess.client("iam").get_role(RoleName=SAGEMAKER_ROLE_NAME)
        info["sagemaker_role_present"] = True
    except ClientError as e:
        info["sagemaker_role_present"] = False
        info["sagemaker_role_error"] = e.response["Error"]["Code"]
    return info


@router.post("/train")
def start_training_job(req: TrainingRequest):
    if not SOURCE_DIR.exists():
        raise HTTPException(500, f"SageMaker source dir not bundled in image: {SOURCE_DIR}")

    bucket = _ensure_bucket()
    role_arn = _resolve_sagemaker_role_arn()

    # Import lazily so the rest of the API still loads if sagemaker SDK is missing
    try:
        import sagemaker
        from sagemaker.pytorch import PyTorch
    except ImportError as e:
        raise HTTPException(500, f"sagemaker SDK not installed: {e}")

    sess = sagemaker.Session(_boto_session())
    job_name = f"sentiment-train-{int(time.time())}"
    output_path = f"s3://{bucket}/sagemaker/output/"
    code_location = f"s3://{bucket}/sagemaker/code/"

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=str(SOURCE_DIR),
        role=role_arn,
        framework_version="2.1.0",
        py_version="py310",
        instance_count=1,
        instance_type=req.instance_type,
        output_path=output_path,
        code_location=code_location,
        sagemaker_session=sess,
        max_run=req.max_run_seconds,
        hyperparameters={
            "model": req.model,
            "epochs": req.epochs,
            "batch-size": req.batch_size,
            "train-samples": req.train_samples,
            "val-samples": req.val_samples,
            "max-len": req.max_len,
            "hf-config": req.hf_config,
        },
        environment={
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        },
    )

    try:
        # wait=False fires-and-forgets so the HTTP call returns quickly
        estimator.fit(job_name=job_name, wait=False)
    except ClientError as e:
        raise HTTPException(500, f"SageMaker submission failed: {e.response['Error']['Message']}")

    return {
        "job_name": job_name,
        "status": "Submitted",
        "instance_type": req.instance_type,
        "output_path": output_path,
        "console_url": (
            f"https://{config.aws.region}.console.aws.amazon.com/sagemaker/home"
            f"?region={config.aws.region}#/jobs/{job_name}"
        ),
    }


@router.get("/jobs")
def list_jobs(limit: int = 20):
    sm = _boto_session().client("sagemaker")
    try:
        resp = sm.list_training_jobs(
            MaxResults=min(max(limit, 1), 50),
            SortBy="CreationTime",
            SortOrder="Descending",
            NameContains="sentiment-train",
        )
    except ClientError as e:
        raise HTTPException(500, e.response["Error"]["Message"])
    jobs: List[dict] = []
    for j in resp.get("TrainingJobSummaries", []):
        jobs.append({
            "name": j["TrainingJobName"],
            "status": j["TrainingJobStatus"],
            "created": j["CreationTime"].isoformat(),
            "ended": j.get("TrainingEndTime").isoformat() if j.get("TrainingEndTime") else None,
        })
    return {"jobs": jobs}


@router.get("/jobs/{job_name}")
def describe_job(job_name: str):
    sm = _boto_session().client("sagemaker")
    try:
        d = sm.describe_training_job(TrainingJobName=job_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            raise HTTPException(404, f"Job {job_name} not found")
        raise HTTPException(500, e.response["Error"]["Message"])

    secondary = [
        {
            "status": s["Status"],
            "started": s["StartTime"].isoformat() if s.get("StartTime") else None,
            "ended": s["EndTime"].isoformat() if s.get("EndTime") else None,
            "message": s.get("StatusMessage"),
        }
        for s in d.get("SecondaryStatusTransitions", [])
    ]
    artifact = d.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    return {
        "name": d["TrainingJobName"],
        "status": d["TrainingJobStatus"],
        "secondary_status": d.get("SecondaryStatus"),
        "failure_reason": d.get("FailureReason"),
        "created": d["CreationTime"].isoformat(),
        "started": d["TrainingStartTime"].isoformat() if d.get("TrainingStartTime") else None,
        "ended": d["TrainingEndTime"].isoformat() if d.get("TrainingEndTime") else None,
        "instance_type": d["ResourceConfig"]["InstanceType"],
        "billable_seconds": d.get("BillableTimeInSeconds"),
        "model_artifact": artifact,
        "transitions": secondary,
        "console_url": (
            f"https://{config.aws.region}.console.aws.amazon.com/sagemaker/home"
            f"?region={config.aws.region}#/jobs/{job_name}"
        ),
    }


@router.get("/jobs/{job_name}/logs")
def job_logs(job_name: str, tail: int = 200):
    """Return the last N lines of CloudWatch logs for a SageMaker training job."""
    logs = _boto_session().client("logs")
    log_group = "/aws/sagemaker/TrainingJobs"
    try:
        streams = logs.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=job_name,
            orderBy="LogStreamName",
            descending=False,
            limit=10,
        ).get("logStreams", [])
    except ClientError as e:
        raise HTTPException(500, e.response["Error"]["Message"])
    if not streams:
        raise HTTPException(404, f"No log streams for {job_name}")

    events: List[dict] = []
    for s in streams:
        try:
            resp = logs.get_log_events(
                logGroupName=log_group,
                logStreamName=s["logStreamName"],
                startFromHead=False,
                limit=tail,
            )
            for ev in resp.get("events", []):
                events.append({
                    "stream": s["logStreamName"],
                    "timestamp": ev["timestamp"],
                    "message": ev["message"],
                })
        except ClientError:
            continue
    events.sort(key=lambda e: e["timestamp"])
    return {"job_name": job_name, "log_group": log_group, "events": events[-tail:]}


@router.post("/jobs/{job_name}/sync")
def sync_artifacts(job_name: str, force: bool = False):
    """Repack a finished SageMaker model.tar.gz into the inference prefix.

    Failed jobs can still have a partial model.tar.gz uploaded (e.g. when one
    of multiple models trained successfully before a later OOM). Pass
    ?force=true to sync those.
    """
    sm = _boto_session().client("sagemaker")
    try:
        d = sm.describe_training_job(TrainingJobName=job_name)
    except ClientError as e:
        raise HTTPException(404, f"Job not found: {e.response['Error']['Message']}")
    status = d["TrainingJobStatus"]
    if status not in ("Completed",) and not force:
        raise HTTPException(409, f"Job is {status}, not Completed. Pass force=true to sync anyway.")
    artifact = d.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not artifact:
        raise HTTPException(409, "Job has no model artifact")

    bucket = config.aws.s3_bucket
    if not artifact.startswith(f"s3://{bucket}/"):
        raise HTTPException(500, f"Artifact not in expected bucket: {artifact}")

    s3 = _boto_session().client("s3")
    key = artifact[len(f"s3://{bucket}/"):]
    uploaded = []
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "model.tar.gz")
        s3.download_file(bucket, key, tar_path)
        out_dir = os.path.join(td, "extracted")
        os.makedirs(out_dir, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(out_dir)
        for fname in os.listdir(out_dir):
            local = os.path.join(out_dir, fname)
            if os.path.isfile(local):
                target = INFERENCE_PREFIX + fname
                s3.upload_file(local, bucket, target)
                uploaded.append(target)
    return {
        "job_name": job_name,
        "uploaded": uploaded,
        "inference_prefix": INFERENCE_PREFIX,
        "note": "Reload the inference service (or restart the task) to pick up new weights.",
    }


@router.post("/jobs/{job_name}/stop")
def stop_job(job_name: str):
    sm = _boto_session().client("sagemaker")
    try:
        sm.stop_training_job(TrainingJobName=job_name)
    except ClientError as e:
        raise HTTPException(500, e.response["Error"]["Message"])
    return {"job_name": job_name, "status": "Stopping"}


@router.post("/reload")
def reload_inference():
    """Drop the cached models so the next request reloads from disk/S3."""
    from app.services import inference_service as svc
    svc._hmgs_model = None
    svc._bilstm_model = None
    return {"status": "reloaded", "note": "Next inference call will re-pull checkpoints."}
