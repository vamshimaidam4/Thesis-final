import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from app.config import config


def get_s3_client():
    """Create S3 client using credentials from environment."""
    return boto3.client(
        "s3",
        aws_access_key_id=config.aws.access_key_id,
        aws_secret_access_key=config.aws.secret_access_key,
        region_name=config.aws.region,
    )


def ensure_bucket_exists():
    """Create the S3 bucket if it doesn't exist."""
    s3 = get_s3_client()
    try:
        s3.head_bucket(Bucket=config.aws.s3_bucket)
        return True
    except ClientError:
        try:
            create_kwargs = {"Bucket": config.aws.s3_bucket}
            if config.aws.region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": config.aws.region
                }
            s3.create_bucket(**create_kwargs)
            return True
        except (ClientError, NoCredentialsError) as e:
            print(f"Could not create bucket: {e}")
            return False


def upload_model_to_s3(local_path: str, s3_key: str) -> bool:
    """Upload a model file to S3."""
    s3 = get_s3_client()
    try:
        s3.upload_file(local_path, config.aws.s3_bucket, s3_key)
        print(f"Uploaded {local_path} to s3://{config.aws.s3_bucket}/{s3_key}")
        return True
    except (ClientError, NoCredentialsError, FileNotFoundError) as e:
        print(f"Upload failed: {e}")
        return False


def download_model_from_s3(s3_key: str, local_path: str) -> bool:
    """Download a model file from S3."""
    s3 = get_s3_client()
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(config.aws.s3_bucket, s3_key, local_path)
        print(f"Downloaded s3://{config.aws.s3_bucket}/{s3_key} to {local_path}")
        return True
    except (ClientError, NoCredentialsError) as e:
        print(f"Download failed: {e}")
        return False


def list_models_in_s3(prefix: str = "models/") -> list:
    """List all model files in the S3 bucket."""
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=config.aws.s3_bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]
    except (ClientError, NoCredentialsError) as e:
        print(f"List failed: {e}")
        return []


def check_aws_connection() -> dict:
    """Verify AWS credentials and connectivity."""
    try:
        s3 = get_s3_client()
        s3.list_buckets()
        return {"status": "connected", "region": config.aws.region, "bucket": config.aws.s3_bucket}
    except NoCredentialsError:
        return {"status": "no_credentials", "error": "AWS credentials not configured"}
    except ClientError as e:
        return {"status": "error", "error": str(e)}
