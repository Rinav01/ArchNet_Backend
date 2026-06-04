import uuid
import time
import random
import logging
from app.tasks.celery_app import celery_app
from app.config.database import SessionLocal
from app.models.training_job import TrainingJob
from app.services.event_dispatcher import EventDispatcher

from app.config.logging import training_logger

logger = training_logger

@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=5,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True
)
def async_run_training_job(self, training_job_id_str: str):
    """Celery background task simulating real-time neural network training on CPU/GPU hardware.
    Updates metrics, loss/accuracy curves, and broadcasts epoch events over Redis Pub/Sub WebSockets.
    """
    job_uuid = uuid.UUID(training_job_id_str)
    db = SessionLocal()
    
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_uuid).first()
        if not job:
            logger.error(f"Training failed: job {training_job_id_str} not found.")
            return {"success": False, "error": "Training job not found."}

        # 1. Update status to RUNNING
        job.status = "RUNNING"
        job.current_epoch = 0
        job.loss_history = []
        job.accuracy_history = []
        db.commit()

        # Publish WebSocket training started event
        EventDispatcher.get_redis().publish(
            "mlbuilder:project:training",
            f'{{"type": "TrainingStarted", "training_job_id": "{training_job_id_str}", "epochs": {job.epochs}}}'
        )

        loss_list = []
        acc_list = []
        
        # Initial starting coordinates for standard convergence curves
        current_loss = 0.85
        current_acc = 0.25

        # 2. Simulate training epochs loop
        for epoch in range(1, job.epochs + 1):
            db.refresh(job)
            if job.status == "CANCELLED":
                logger.info(f"Training job {training_job_id_str} was cancelled.")
                return {"success": False, "error": "Job cancelled by user."}

            time.sleep(1.0) # Simulate hardware operational compute delays
            
            db.refresh(job)
            if job.status == "CANCELLED":
                logger.info(f"Training job {training_job_id_str} was cancelled.")
                return {"success": False, "error": "Job cancelled by user."}
            
            # Synthesize convergence increments: loss decays, accuracy rises
            decay_rate = random.uniform(0.05, 0.15)
            growth_rate = random.uniform(0.04, 0.12)
            
            current_loss = max(0.02, current_loss * (1 - decay_rate))
            current_acc = min(0.99, current_acc + (1 - current_acc) * growth_rate)

            loss_list.append(round(current_loss, 4))
            acc_list.append(round(current_acc, 4))

            # Commit incremental stats
            job.current_epoch = epoch
            job.loss_history = loss_list
            job.accuracy_history = acc_list
            db.commit()

            # Broadcast WebSocket progressive telemetry coordinates
            EventDispatcher.get_redis().publish(
                "mlbuilder:project:training",
                f'{{"type": "TrainingEpochProgress", "training_job_id": "{training_job_id_str}", "current_epoch": {epoch}, "loss": {current_loss:.4f}, "accuracy": {current_acc:.4f}}}'
            )

        # 3. Training completed successfully!
        job.status = "COMPLETED"
        job.metrics_metadata = {
            "device": "CUDA (NVIDIA T4)" if random.choice([True, False]) else "CPU (Intel Xeon)",
            "peak_memory_used_mb": random.randint(120, 480),
            "training_duration_seconds": job.epochs * 1,
            "final_loss": loss_list[-1],
            "final_accuracy": acc_list[-1],
            "logs": f"Epochs finished successfully. Hardware utilized correctly. Mode: Local worker execution."
        }
        db.commit()

        # Broadcast completed event
        EventDispatcher.get_redis().publish(
            "mlbuilder:project:training",
            f'{{"type": "TrainingCompleted", "training_job_id": "{training_job_id_str}", "status": "COMPLETED", "accuracy": {acc_list[-1]}}}'
        )

        return {
            "success": True,
            "training_job_id": training_job_id_str,
            "final_loss": loss_list[-1],
            "final_accuracy": acc_list[-1]
        }

    except Exception as e:
        logger.error(f"Training loop crashed: {e}", exc_info=True)
        db.rollback()
        try:
            if self.request.retries < self.max_retries:
                logger.info(f"Retrying training task {self.request.id} ({self.request.retries + 1}/{self.max_retries})")
                self.retry(exc=e)
            else:
                failed_job = db.query(TrainingJob).filter(TrainingJob.id == job_uuid).first()
                if failed_job:
                    failed_job.status = "FAILED"
                    failed_job.metrics_metadata = {"error": str(e), "logs": "Crash logs recorded after max retries."}
                    db.commit()

                EventDispatcher.get_redis().publish(
                    "mlbuilder:project:training",
                    f'{{"type": "TrainingFailed", "training_job_id": "{training_job_id_str}", "status": "FAILED", "error": "{str(e)}"}}'
                )
        except Exception as rb_err:
            logger.error(f"Training rollback database fail: {rb_err}")

        return {"success": False, "error": str(e)}
    finally:
        db.close()
