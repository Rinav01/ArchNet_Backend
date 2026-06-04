import os
import uuid
import shutil
import logging
from app.config.settings import settings

logger = logging.getLogger("mlbuilder.dataset_storage")

class DatasetStorage:
    @staticmethod
    def get_storage_path(dataset_id: uuid.UUID, filename: str) -> str:
        """Determines the target storage path based on configurations (S3, GCS, or Local)."""
        aws_key = settings.AWS_ACCESS_KEY_ID
        aws_secret = settings.AWS_SECRET_ACCESS_KEY
        bucket = settings.AWS_BUCKET_NAME
        gcp_bucket = settings.GCP_BUCKET_NAME

        if gcp_bucket:
            return f"gs://{gcp_bucket}/datasets/{str(dataset_id)}/{filename}"
        elif aws_key and aws_secret and bucket:
            return f"s3://{bucket}/datasets/{str(dataset_id)}/{filename}"
        else:
            # Local Storage Path in scratch/storage
            workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            path = os.path.join(workspace_dir, "scratch", "storage", str(dataset_id), filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path

    @staticmethod
    def save_local_file(source_path: str, dataset_id: uuid.UUID, filename: str) -> str:
        """Saves a local file into the dataset storage directory."""
        target_path = DatasetStorage.get_storage_path(dataset_id, filename)
        if not target_path.startswith("s3://") and not target_path.startswith("gs://"):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(source_path, target_path)
            logger.info(f"Saved dataset file locally to {target_path}")
        return target_path

    @staticmethod
    def delete_dataset_file(storage_path: str) -> None:
        """Deletes the stored dataset file if it is local, otherwise handles remote deletes."""
        if not storage_path:
            return

        if storage_path.startswith("s3://"):
            try:
                import boto3
                from urllib.parse import urlparse
                parsed = urlparse(storage_path)
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
                )
                s3.delete_object(Bucket=bucket, Key=key)
                logger.info(f"Deleted S3 dataset blob: {storage_path}")
            except Exception as e:
                logger.error(f"Failed to delete S3 dataset file {storage_path}: {e}")
        elif storage_path.startswith("gs://"):
            try:
                import json
                from google.cloud import storage
                from google.oauth2 import service_account
                from urllib.parse import urlparse
                
                parsed = urlparse(storage_path)
                bucket_name = parsed.netloc
                blob_name = parsed.path.lstrip("/")
                
                if settings.GCP_CREDENTIALS_JSON:
                    creds_dict = json.loads(settings.GCP_CREDENTIALS_JSON)
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    storage_client = storage.Client(credentials=credentials, project=settings.GCP_PROJECT_ID)
                else:
                    storage_client = storage.Client()
                    
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.delete()
                logger.info(f"Deleted GCS dataset blob: {storage_path}")
            except Exception as e:
                logger.error(f"Failed to delete GCS dataset file {storage_path}: {e}")
        else:
            if os.path.exists(storage_path):
                try:
                    os.remove(storage_path)
                    parent_dir = os.path.dirname(storage_path)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                    logger.info(f"Deleted local dataset file: {storage_path}")
                except Exception as e:
                    logger.error(f"Failed to delete local dataset file {storage_path}: {e}")
