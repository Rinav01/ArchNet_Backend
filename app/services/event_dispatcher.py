import json
import uuid
import redis
from typing import List, Dict, Any
from app.config.settings import settings

class EventDispatcher:
    _redis_client = None

    @classmethod
    def get_redis(cls):
        if cls._redis_client is None:
            cls._redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        return cls._redis_client

    @classmethod
    def _publish(cls, project_id: uuid.UUID, event: Dict[str, Any]):
        """Serializes and publishes an event message payload to Redis Pub/Sub channel."""
        try:
            r = cls.get_redis()
            channel = f"mlbuilder:project:{str(project_id)}"
            r.publish(channel, json.dumps(event))
        except Exception as e:
            # Silently log/fallback if redis is not running (e.g. inside unit tests without broker)
            print(f"[EventDispatcher Warning] Redis publish failed: {e}")

    @classmethod
    def dispatch_node_added(cls, project_id: uuid.UUID, node_id: uuid.UUID, label: str, node_type: str):
        event = {
            "type": "NodeAdded",
            "project_id": str(project_id),
            "node_id": str(node_id),
            "label": label,
            "node_type": node_type
        }
        cls._publish(project_id, event)

    @classmethod
    def dispatch_node_deleted(cls, project_id: uuid.UUID, node_id: uuid.UUID):
        event = {
            "type": "NodeDeleted",
            "project_id": str(project_id),
            "node_id": str(node_id)
        }
        cls._publish(project_id, event)

    @classmethod
    def dispatch_compilation_started(cls, project_id: uuid.UUID, task_id: str):
        event = {
            "type": "CompilationStarted",
            "project_id": str(project_id),
            "task_id": task_id
        }
        cls._publish(project_id, event)

    @classmethod
    def dispatch_compilation_finished(cls, project_id: uuid.UUID, code: str, logs: str):
        event = {
            "type": "CompilationFinished",
            "project_id": str(project_id),
            "code": code,
            "logs": logs
        }
        cls._publish(project_id, event)

    @classmethod
    def dispatch_validation_failed(cls, project_id: uuid.UUID, errors: List[str]):
        event = {
            "type": "ValidationFailed",
            "project_id": str(project_id),
            "errors": errors
        }
        cls._publish(project_id, event)
