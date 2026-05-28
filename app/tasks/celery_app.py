from celery import Celery
from app.config.settings import settings

celery_app = Celery(
    "mlbuilder",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.tasks", "app.tasks.dataset_tasks", "app.tasks.training_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30,  # 30-second absolute time limit for async compile operations
)
