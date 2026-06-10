import uuid
import logging
from app.config.settings import settings

logger = logging.getLogger("mlbuilder.storage_service")

class S3Service:
    @staticmethod
    def generate_presigned_upload_url(dataset_id: uuid.UUID, filename: str) -> str:
        """Generates a secure pre-signed PUT upload URL.
        Supports AWS S3, Google Cloud Storage (GCS), or Local mock loopbacks
        based on active configurations in environment settings.
        """
        aws_key = settings.AWS_ACCESS_KEY_ID
        aws_secret = settings.AWS_SECRET_ACCESS_KEY
        aws_bucket = settings.AWS_BUCKET_NAME

        gcp_bucket = settings.GCP_BUCKET_NAME
        gcp_creds = settings.GCP_CREDENTIALS_JSON

        # 1. Google Cloud Storage (GCS) Pre-signed PUT Generation
        if gcp_bucket and gcp_creds:
            try:
                import json
                from google.cloud import storage
                from google.oauth2 import service_account

                creds_dict = json.loads(gcp_creds)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                storage_client = storage.Client(credentials=credentials, project=settings.GCP_PROJECT_ID)

                bucket = storage_client.bucket(gcp_bucket)
                blob_name = f"datasets/{str(dataset_id)}/{filename}"
                blob = bucket.blob(blob_name)

                url = blob.generate_signed_url(
                    version="v4",
                    expiration=3600,
                    method="PUT",
                    content_type="application/octet-stream"
                )
                logger.info(f"Generated Google Cloud Storage signed URL for blob: {blob_name}")
                return url
            except Exception as e:
                logger.error(f"Failed to generate Google Cloud Storage signed URL: {e}. Falling back to AWS/Local.")

        # 2. AWS S3 Pre-signed PUT Generation
        if aws_key and aws_secret and aws_bucket:
            try:
                import boto3
                from botocore.config import Config

                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=aws_key,
                    aws_secret_access_key=aws_secret,
                    config=Config(signature_version="s3v4")
                )

                s3_key = f"datasets/{str(dataset_id)}/{filename}"
                url = s3_client.generate_presigned_url(
                    "put_object",
                    Params={
                        "Bucket": aws_bucket,
                        "Key": s3_key,
                        "ContentType": "application/octet-stream"
                    },
                    ExpiresIn=3600
                )
                logger.info(f"Generated AWS S3 pre-signed URL for key: {s3_key}")
                return url
            except Exception as e:
                logger.error(f"Failed to generate AWS S3 pre-signed URL: {e}. Falling back to local uploader.")

        # 3. Local Developer Fallback Mock Link (FastAPI route)
        local_url = f"http://localhost:8000/api/storage/upload/{str(dataset_id)}?filename={filename}"
        logger.info(f"Generated local mock uploader link: {local_url}")
        return local_url

    @staticmethod
    def generate_presigned_download_url(artifact_path: str) -> str:
        """Generates a secure pre-signed GET download URL for an artifact path."""
        import os
        aws_key = settings.AWS_ACCESS_KEY_ID
        aws_secret = settings.AWS_SECRET_ACCESS_KEY
        aws_bucket = settings.AWS_BUCKET_NAME

        gcp_bucket = settings.GCP_BUCKET_NAME
        gcp_creds = settings.GCP_CREDENTIALS_JSON

        if artifact_path.startswith("gs://"):
            if gcp_bucket and gcp_creds:
                try:
                    import json
                    from google.cloud import storage
                    from google.oauth2 import service_account

                    # gs://bucket/path/to/file -> bucket: bucket, blob: path/to/file
                    path_parts = artifact_path[5:].split("/", 1)
                    b_name = path_parts[0]
                    blob_name = path_parts[1] if len(path_parts) > 1 else ""

                    creds_dict = json.loads(gcp_creds)
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    storage_client = storage.Client(credentials=credentials, project=settings.GCP_PROJECT_ID)

                    bucket = storage_client.bucket(b_name)
                    blob = bucket.blob(blob_name)

                    url = blob.generate_signed_url(
                        version="v4",
                        expiration=3600,
                        method="GET"
                    )
                    return url
                except Exception as e:
                    logger.error(f"Failed to generate GCS download URL: {e}")
            
        elif artifact_path.startswith("s3://"):
            if aws_key and aws_secret and aws_bucket:
                try:
                    import boto3
                    from botocore.config import Config

                    path_parts = artifact_path[5:].split("/", 1)
                    b_name = path_parts[0]
                    s3_key = path_parts[1] if len(path_parts) > 1 else ""

                    s3_client = boto3.client(
                        "s3",
                        aws_access_key_id=aws_key,
                        aws_secret_access_key=aws_secret,
                        config=Config(signature_version="s3v4")
                    )

                    url = s3_client.generate_presigned_url(
                        "get_object",
                        Params={
                            "Bucket": b_name,
                            "Key": s3_key
                        },
                        ExpiresIn=3600
                    )
                    return url
                except Exception as e:
                    logger.error(f"Failed to generate S3 download URL: {e}")

        # Fallback for local files
        filename = os.path.basename(artifact_path)
        return f"http://localhost:8000/exports/{filename}"

