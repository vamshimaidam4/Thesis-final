"""
AWS deployment utilities for the sentiment analysis project.
Handles S3 bucket setup, model upload/download, and EC2 instance management.

Usage:
    python aws/deploy.py setup       # Create S3 bucket and upload models
    python aws/deploy.py download     # Download models from S3
    python aws/deploy.py status       # Check AWS connectivity
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def get_config():
    return {
        "access_key": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        "bucket": os.getenv("S3_BUCKET_NAME", "deep-learning-sentiment-models"),
    }


def get_s3_client(cfg):
    return boto3.client(
        "s3",
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        region_name=cfg["region"],
    )


def check_status(cfg):
    """Check AWS connectivity and list resources."""
    print("Checking AWS connection...")
    try:
        s3 = get_s3_client(cfg)
        buckets = s3.list_buckets()
        print(f"Connected to AWS ({cfg['region']})")
        print(f"Buckets: {[b['Name'] for b in buckets['Buckets']]}")

        try:
            s3.head_bucket(Bucket=cfg["bucket"])
            print(f"Model bucket '{cfg['bucket']}' exists")

            objects = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="models/")
            if "Contents" in objects:
                print("Models in S3:")
                for obj in objects["Contents"]:
                    size_mb = obj["Size"] / (1024 * 1024)
                    print(f"  {obj['Key']} ({size_mb:.1f} MB)")
            else:
                print("No models uploaded yet")
        except ClientError:
            print(f"Bucket '{cfg['bucket']}' does not exist yet")

        # Check EC2 instances
        ec2 = boto3.client(
            "ec2",
            aws_access_key_id=cfg["access_key"],
            aws_secret_access_key=cfg["secret_key"],
            region_name=cfg["region"],
        )
        instances = ec2.describe_instances(
            Filters=[{"Name": "tag:Project", "Values": ["sentiment-analysis"]}]
        )
        running = []
        for res in instances["Reservations"]:
            for inst in res["Instances"]:
                if inst["State"]["Name"] in ("running", "pending"):
                    running.append(inst["InstanceId"])
        if running:
            print(f"Running EC2 instances: {running}")
        else:
            print("No project EC2 instances running")

        return True
    except NoCredentialsError:
        print("ERROR: AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def setup_s3(cfg):
    """Create S3 bucket and upload model files."""
    s3 = get_s3_client(cfg)

    # Create bucket
    try:
        s3.head_bucket(Bucket=cfg["bucket"])
        print(f"Bucket '{cfg['bucket']}' already exists")
    except ClientError:
        print(f"Creating bucket '{cfg['bucket']}'...")
        create_kwargs = {"Bucket": cfg["bucket"]}
        if cfg["region"] != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": cfg["region"]
            }
        s3.create_bucket(**create_kwargs)
        print("Bucket created")

    # Upload models
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    if os.path.isdir(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith((".pt", ".pth", ".bin", ".json")):
                local_path = os.path.join(model_dir, filename)
                s3_key = f"models/{filename}"
                print(f"Uploading {filename}...")
                s3.upload_file(local_path, cfg["bucket"], s3_key)
                print(f"  Uploaded to s3://{cfg['bucket']}/{s3_key}")
    else:
        print("No local models directory found. Train models first.")


def download_models(cfg):
    """Download models from S3."""
    s3 = get_s3_client(cfg)
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    try:
        objects = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="models/")
        if "Contents" not in objects:
            print("No models found in S3")
            return

        for obj in objects["Contents"]:
            filename = os.path.basename(obj["Key"])
            local_path = os.path.join(model_dir, filename)
            print(f"Downloading {obj['Key']}...")
            s3.download_file(cfg["bucket"], obj["Key"], local_path)
            print(f"  Saved to {local_path}")
    except ClientError as e:
        print(f"Download error: {e}")


def main():
    parser = argparse.ArgumentParser(description="AWS deployment utilities")
    parser.add_argument("action", choices=["setup", "download", "status"])
    args = parser.parse_args()

    cfg = get_config()

    if not cfg["access_key"] or not cfg["secret_key"]:
        print("WARNING: AWS credentials not set in .env file")
        if args.action != "status":
            return

    actions = {
        "status": lambda: check_status(cfg),
        "setup": lambda: setup_s3(cfg),
        "download": lambda: download_models(cfg),
    }
    actions[args.action]()


if __name__ == "__main__":
    main()
