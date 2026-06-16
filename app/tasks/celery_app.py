from celery import Celery
from kombu import Queue
from app.config.settings import settings

celery_app = Celery(
    "mlbuilder",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.tasks", "app.tasks.dataset_tasks", "app.tasks.training_tasks", "app.workers.training_worker"]
)

celery_app.conf.update(
    task_always_eager=settings.ENVIRONMENT == "development",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30,  # 30-second absolute time limit for async compile operations
    task_queues=(
        Queue("high_priority", routing_key="high_priority"),
        Queue("low_priority", routing_key="low_priority"),
    ),
    task_default_queue="high_priority",
    task_routes={
        "app.tasks.dataset_tasks.async_process_dataset": {"queue": "high_priority"},
        "app.tasks.tasks.async_compile_and_validate": {"queue": "high_priority"},
        "app.tasks.tasks.async_dataset_verification_preflight": {"queue": "high_priority"},
        "app.tasks.training_tasks.async_run_training_job": {"queue": "low_priority"},
        "app.workers.training_worker.async_run_training_pipeline": {"queue": "low_priority"},
    }
)
