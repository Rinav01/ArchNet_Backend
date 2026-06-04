import os
import uuid
import logging
from app.tasks.celery_app import celery_app
from app.config.database import SessionLocal
from app.models.dataset import Dataset
from app.services.dataset_parsers import DatasetParserService
from app.services.event_dispatcher import EventDispatcher
from app.config.settings import settings

logger = logging.getLogger("mlbuilder.dataset_tasks")

@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=5,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True
)
def async_process_dataset(self, dataset_id_str: str):
    """Celery task executing deep schema parsing and Pillow image analytics in the background.
    Publishes real-time Redis/WebSocket channel signals.
    """
    dataset_uuid = uuid.UUID(dataset_id_str)
    
    # 1. Dispatch "DatasetProcessing" WebSocket signal
    db = SessionLocal()
    temp_local_file = None
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
        if not dataset:
            logger.error(f"Ingestion failed: dataset '{dataset_id_str}' not found.")
            return {"success": False, "error": "Dataset not found."}

        # Update status to processing
        dataset.status = "PROCESSING"
        db.commit()

        # Publish WebSocket progress event
        EventDispatcher.get_redis().publish(
            "mlbuilder:project:dataset",  # global dataset channel or specific user channel
            f'{{"type": "DatasetProcessing", "dataset_id": "{dataset_id_str}"}}'
        )

        file_path = dataset.file_path
        if not file_path:
            raise ValueError("Dataset file path is missing.")

        # If stored on S3, download to a local temporary location in workspace scratch folder
        if file_path.startswith("s3://"):
            import boto3
            from urllib.parse import urlparse
            
            parsed_url = urlparse(file_path)
            bucket_name = parsed_url.netloc
            s3_key = parsed_url.path.lstrip("/")

            workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            scratch_temp = os.path.join(workspace_dir, "scratch", "temp")
            os.makedirs(scratch_temp, exist_ok=True)
            
            temp_local_file = os.path.join(scratch_temp, f"s3_temp_{str(dataset_uuid)}_{os.path.basename(file_path)}")
            
            logger.info(f"Downloading dataset from S3: {file_path} -> {temp_local_file}")
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            s3_client.download_file(bucket_name, s3_key, temp_local_file)
            file_path = temp_local_file

        # 2. Trigger appropriate parsing engine
        from app.services.dataset_analyzer import DatasetAnalyzer
        analysis = DatasetAnalyzer.analyze_dataset(file_path, dataset.dataset_type)
        row_count = analysis["row_count"]
        column_count = analysis["column_count"]
        metadata_json = analysis["metadata_json"]

        # 3. Success! Commit schema mappings
        dataset.row_count = row_count
        dataset.column_count = column_count
        dataset.metadata_json = metadata_json
        dataset.status = "READY"
        db.commit()

        # Publish finished event over Redis Pub/Sub
        EventDispatcher.get_redis().publish(
            "mlbuilder:project:dataset",
            f'{{"type": "DatasetProcessed", "dataset_id": "{dataset_id_str}", "status": "READY", "num_records": {row_count}}}'
        )

        return {
            "success": True,
            "dataset_id": dataset_id_str,
            "num_records": row_count,
            "metadata": metadata_json
        }

    except Exception as e:
        logger.error(f"Dataset parsing crashed: {e}", exc_info=True)
        db.rollback()
        try:
            if self.request.retries < self.max_retries:
                logger.info(f"Retrying dataset task {self.request.id} ({self.request.retries + 1}/{self.max_retries})")
                self.retry(exc=e)
            else:
                failed_dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
                if failed_dataset:
                    failed_dataset.status = "FAILED"
                    db.commit()
                EventDispatcher.get_redis().publish(
                    "mlbuilder:project:dataset",
                    f'{{"type": "DatasetFailed", "dataset_id": "{dataset_id_str}", "status": "FAILED", "error": "{str(e)}"}}'
                )
        except Exception as rollback_err:
            logger.error(f"Ingestion database rollback fail: {rollback_err}")
            
        return {"success": False, "error": str(e)}
    finally:
        db.close()
        # Clean up local temporary S3 file if it was created
        if temp_local_file and os.path.exists(temp_local_file):
            try:
                os.remove(temp_local_file)
                logger.info(f"Cleaned up temp S3 file: {temp_local_file}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up temp S3 file: {cleanup_err}")

