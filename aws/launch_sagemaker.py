"""Launch a SageMaker training job for HMGS + BiLSTM.

Run from CI (or any host with AWS credentials):

    python aws/launch_sagemaker.py \
        --instance-type ml.g4dn.xlarge \
        --epochs 3 --train-samples 10000

The script:
  1. Ensures the S3 bucket and SageMaker execution role exist (creates them
     if missing).
  2. Submits a PyTorch training job using the source dir aws/sagemaker_train.
  3. Waits for completion and prints the S3 path to model.tar.gz.
  4. Optionally extracts the tarball and re-uploads checkpoints to a stable
     prefix that the inference backend looks up at startup.

Inputs come from environment variables (set as GitHub Actions secrets):
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION (optional)
  S3_BUCKET_NAME (optional, defaults to deep-learning-sentiment-models)
"""
import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROLE_NAME = "SageMakerSentimentExecutionRole"
DEFAULT_BUCKET = os.getenv("S3_BUCKET_NAME", "deep-learning-sentiment-models")
DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
INFERENCE_PREFIX = "inference/current/"  # stable prefix the backend reads


def ensure_bucket(s3, bucket, region):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"[s3] bucket exists: {bucket}", flush=True)
    except ClientError:
        kwargs = {"Bucket": bucket}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**kwargs)
        print(f"[s3] created bucket: {bucket}", flush=True)


def ensure_role(iam):
    try:
        role = iam.get_role(RoleName=ROLE_NAME)
        return role["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }
    iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="SageMaker execution role for sentiment-analysis training jobs",
    )
    for arn in (
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
    ):
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=arn)
    print(f"[iam] created role {ROLE_NAME}", flush=True)
    # IAM is eventually consistent; give SageMaker a moment.
    time.sleep(15)
    return iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]


def submit_job(role_arn, args):
    import sagemaker
    from sagemaker.pytorch import PyTorch

    session = sagemaker.Session(boto3.Session(region_name=args.region))
    job_name = f"sentiment-train-{int(time.time())}"
    output_path = f"s3://{args.bucket}/sagemaker/output/"
    code_location = f"s3://{args.bucket}/sagemaker/code/"

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=str(Path(__file__).parent / "sagemaker_train"),
        role=role_arn,
        framework_version="2.1.0",
        py_version="py310",
        instance_count=1,
        instance_type=args.instance_type,
        output_path=output_path,
        code_location=code_location,
        sagemaker_session=session,
        max_run=args.max_run_seconds,
        hyperparameters={
            "model": args.model,
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "train-samples": args.train_samples,
            "val-samples": args.val_samples,
            "max-len": args.max_len,
            "hf-config": args.hf_config,
        },
        environment={
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        },
    )

    print(f"[sagemaker] submitting job {job_name} on {args.instance_type}", flush=True)
    estimator.fit(job_name=job_name, wait=True, logs="All")
    return estimator, job_name


def repack_for_inference(s3, bucket, model_data_url):
    """Download SageMaker output tarball, extract checkpoints, upload to inference prefix."""
    print(f"[repack] fetching {model_data_url}", flush=True)
    assert model_data_url.startswith(f"s3://{bucket}/"), model_data_url
    key = model_data_url[len(f"s3://{bucket}/"):]
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
                print(f"[repack] s3://{bucket}/{target}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-type", default="ml.g4dn.xlarge",
                   help="ml.g4dn.xlarge ($0.74/hr) is a good default. "
                        "Use ml.m5.xlarge ($0.23/hr CPU-only) for cheap smoke tests.")
    p.add_argument("--model", choices=["hmgs", "bilstm", "both"], default="both")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-samples", type=int, default=10000)
    p.add_argument("--val-samples", type=int, default=1000)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--hf-config", default="raw_review_All_Beauty",
                   help="Subset of McAuley-Lab/Amazon-Reviews-2023 to use")
    p.add_argument("--max-run-seconds", type=int, default=4 * 3600)
    p.add_argument("--no-repack", action="store_true",
                   help="Skip the S3 repack step (artifacts stay in sagemaker/output/)")
    args = p.parse_args()

    boto = boto3.Session(region_name=args.region)
    s3 = boto.client("s3")
    iam = boto.client("iam")

    ensure_bucket(s3, args.bucket, args.region)
    role_arn = ensure_role(iam)
    print(f"[iam] role arn: {role_arn}", flush=True)

    estimator, job_name = submit_job(role_arn, args)
    model_data_url = estimator.model_data
    print(f"[done] job={job_name}", flush=True)
    print(f"[done] model artifact: {model_data_url}", flush=True)

    if not args.no_repack:
        repack_for_inference(s3, args.bucket, model_data_url)
        print(f"[done] inference checkpoints at s3://{args.bucket}/{INFERENCE_PREFIX}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
