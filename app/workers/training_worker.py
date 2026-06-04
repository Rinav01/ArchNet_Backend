import uuid
import logging
from celery import shared_task
from app.tasks.celery_app import celery_app
from app.config.database import SessionLocal
from app.models.training_job import TrainingJob
from app.services.training_service import TrainingService
from app.services.event_dispatcher import EventDispatcher
from app.config.logging import training_logger

logger = training_logger

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    name="app.workers.training_worker.async_run_training_pipeline"
)
def async_run_training_pipeline(self, payload: dict):
    """
    Celery background task executing deep training loops on CPU/GPU hardware.
    Consumes dictionary payload:
    {
        "projectId": "...",
        "datasetId": "...",
        "epochs": 20,
        "trainingJobId": "..."  # optional
    }
    Updates status, logs loss/accuracy convergence curves, and pushes Redis updates.
    """
    logger.info(f"Worker received training payload: {payload}")
    db = SessionLocal()

    try:
        # 1. Parse IDs from payload
        proj_id_str = payload.get("projectId")
        dataset_id_str = payload.get("datasetId")
        epochs = int(payload.get("epochs", 10))
        job_id_str = payload.get("trainingJobId")

        if not proj_id_str:
            raise ValueError("Missing 'projectId' in payload.")

        project_uuid = uuid.UUID(proj_id_str)
        dataset_uuid = uuid.UUID(dataset_id_str) if dataset_id_str else None

        # 2. Retrieve or Create Training Job Record
        job = None
        if job_id_str:
            job_uuid = uuid.UUID(job_id_str)
            job = db.query(TrainingJob).filter(TrainingJob.id == job_uuid).first()

        if not job:
            # Fallback: look for latest PENDING training job
            job = db.query(TrainingJob).filter(
                TrainingJob.project_id == project_uuid,
                TrainingJob.status == "PENDING"
            ).order_by(TrainingJob.created_at.desc()).first()

        if not job:
            # Create a new job if none exists
            job_uuid = uuid.UUID(job_id_str) if job_id_str else uuid.uuid4()
            job = TrainingJob(
                id=job_uuid,
                project_id=project_uuid,
                dataset_id=dataset_uuid,
                status="PENDING",
                epochs=epochs,
                current_epoch=0,
                loss_history=[],
                accuracy_history=[],
                metrics_metadata={}
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            logger.info(f"Created new TrainingJob record: {job.id}")
        else:
            job_uuid = job.id

        # Bind celery task ID
        job.celery_task_id = self.request.id
        db.commit()

        # 3. Trigger Service pipeline
        result = TrainingService.run_pipeline(
            db=db,
            project_id=project_uuid,
            dataset_id=dataset_uuid,
            epochs=epochs,
            training_job_id=job_uuid
        )

        return result

    except Exception as e:
        logger.error(f"Worker training task failed: {e}", exc_info=True)
        db.rollback()
        
        # Publish task execution failure event
        if "job_uuid" in locals():
            try:
                failed_job = db.query(TrainingJob).filter(TrainingJob.id == job_uuid).first()
                if failed_job and failed_job.status != "CANCELLED":
                    failed_job.status = "FAILED"
                    failed_job.metrics_metadata = {
                        "error": str(e),
                        "logs": f"Worker task crashed: {str(e)}"
                    }
                    db.commit()
            except Exception as db_err:
                logger.error(f"Failed to save failed status in worker crash cleanup: {db_err}")

            EventDispatcher.get_redis().publish(
                "mlbuilder:project:training",
                f'{{"type": "TrainingFailed", "training_job_id": "{str(job_uuid)}", "status": "FAILED", "error": "{str(e)}"}}'
            )

        # Retry if database locks or temporary connection issue
        try:
            if self.request.retries < self.max_retries:
                logger.info(f"Retrying worker training task ({self.request.retries + 1}/{self.max_retries})")
                self.retry(exc=e)
        except Exception as retry_err:
            logger.error(f"Retry scheduling crashed: {retry_err}")

        return {"success": False, "error": str(e)}

    finally:
        db.close()
