import redis
import logging
import uuid
from typing import Any, Optional
from app.config.settings import settings

logger = logging.getLogger("mlbuilder.cache")

class CachingService:
    _redis_client = None

    @classmethod
    def get_redis(cls):
        if cls._redis_client is None:
            cls._redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        return cls._redis_client

    @classmethod
    def get(cls, key: str) -> Optional[str]:
        """Fetches a string value from Redis cache."""
        try:
            r = cls.get_redis()
            return r.get(key)
        except Exception as e:
            logger.error(f"Redis cache GET error for key '{key}': {e}")
            return None

    @classmethod
    def set(cls, key: str, value: str, expire_seconds: int = 3600) -> None:
        """Saves a string value to Redis cache with custom expiration."""
        try:
            r = cls.get_redis()
            r.set(key, value, ex=expire_seconds)
        except Exception as e:
            logger.error(f"Redis cache SET error for key '{key}': {e}")

    @classmethod
    def delete(cls, key: str) -> None:
        """Deletes a key from Redis cache."""
        try:
            r = cls.get_redis()
            r.delete(key)
        except Exception as e:
            logger.error(f"Redis cache DELETE error for key '{key}': {e}")

    @classmethod
    def invalidate_project_cache(cls, project_id: Any) -> None:
        """Invalidates all cached topologies and compiled scripts associated with a project canvas."""
        try:
            pid = str(project_id)
            r = cls.get_redis()
            keys_to_delete = [
                f"cache:project:nodes:{pid}",
                f"cache:project:edges:{pid}",
                f"cache:project:pytorch:{pid}",
                f"cache:project:tensorflow:{pid}",
                f"cache:project:jax:{pid}",
                f"cache:project:onnx:{pid}",
                f"cache:project:automl:{pid}",
                f"cache:project:benchmark:{pid}"
            ]
            logger.info(f"Invalidating cache keys for project '{pid}'")
            r.delete(*keys_to_delete)
        except Exception as e:
            logger.error(f"Failed to invalidate cache keys for project '{project_id}': {e}")
