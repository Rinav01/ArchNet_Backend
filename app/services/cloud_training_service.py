import os
import json
import uuid
import logging
from sqlalchemy.orm import Session
from app.config.settings import settings
from app.models.training_job import TrainingJob
from app.tasks.training_tasks import async_run_training_job

logger = logging.getLogger("mlbuilder.cloud_training_service")

class CloudTrainingService:
    @staticmethod
    def trigger_training_job(
        db: Session,
        project_id: uuid.UUID,
        epochs: int,
        dataset_id: uuid.UUID | None = None
    ) -> tuple[TrainingJob, str]:
        """Creates a TrainingJob database record and routes execution:
        Triggers a managed Google Cloud Vertex AI CustomJob if credentials exist,
        otherwise falls back to scheduling the Celery local training task loop.
        """
        job_id = uuid.uuid4()
        
        # Determine GCP active config
        gcp_bucket = settings.GCP_BUCKET_NAME
        gcp_creds = settings.GCP_CREDENTIALS_JSON

        new_job = TrainingJob(
            id=job_id,
            project_id=project_id,
            dataset_id=dataset_id,
            status="PENDING",
            epochs=epochs,
            current_epoch=0,
            loss_history=[],
            accuracy_history=[],
            metrics_metadata={}
        )
        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        # 1. Google Cloud Vertex AI Custom Job Triggering
        if gcp_bucket and gcp_creds:
            try:
                from google.cloud import aiplatform
                from google.oauth2 import service_account

                creds_dict = json.loads(gcp_creds)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                
                # Initialize Vertex AI SDK
                aiplatform.init(
                    project=settings.GCP_PROJECT_ID,
                    location="us-central1",
                    credentials=credentials
                )

                # Script parameters to pass as arguments to PyTorch container
                gcs_dataset_uri = f"gs://{gcp_bucket}/datasets/{str(dataset_id)}" if dataset_id else ""
                
                # Trigger a Managed Custom Container Training Job on Vertex AI
                # We point to standard pre-built Google PyTorch GPU container image
                custom_job = aiplatform.CustomJob(
                    display_name=f"mlbuilder-training-{str(job_id)}",
                    worker_pool_specs=[{
                        "machine_spec": {
                            "machine_type": "n1-standard-4",
                            "accelerator_type": "NVIDIA_TESLA_T4",
                            "accelerator_count": 1
                        },
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
                            "args": [
                                "--training_job_id", str(job_id),
                                "--epochs", str(epochs),
                                "--dataset_uri", gcs_dataset_uri,
                                "--webhook_url", "http://localhost:8000/api/training/gcp-webhook"
                            ]
                        }
                    }]
                )
                
                # Execute asynchronously to avoid locking FastAPI thread
                custom_job.run(sync=False)
                
                # Update training job status and metadata to mark Google Vertex integration
                new_job.metrics_metadata = {
                    "provider": "Google Cloud Vertex AI",
                    "machine_type": "n1-standard-4",
                    "accelerator": "NVIDIA Tesla T4 (1x)",
                    "custom_job_id": custom_job.resource_name,
                    "logs": "Vertex AI CustomJob triggered and running asynchronously."
                }
                db.commit()

                logger.info(f"Successfully triggered Vertex AI job: {custom_job.resource_name}")
                return new_job, f"vertex-ai-job-{str(job_id)}"
                
            except Exception as e:
                logger.error(f"Failed to launch GCP Vertex AI custom job: {e}. Falling back to Celery local training.")
                # Fall back gracefully to Celery task if Vertex API triggers fail
                task = async_run_training_job.delay(str(job_id))
                return new_job, task.id

        # 2. Local Celery fallback loop
        task = async_run_training_job.delay(str(job_id))
        return new_job, task.id
