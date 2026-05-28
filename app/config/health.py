import logging
import redis
from fastapi import APIRouter, Response, status
from sqlalchemy.sql import text
from app.config.database import SessionLocal
from app.config.settings import settings

logger = logging.getLogger("mlbuilder.health")
health_router = APIRouter()

@health_router.get("/health")
def health_check(response: Response):
    """Deep health check endpoint validating PostgreSQL and Redis connectivity.
    Returns 200 OK if both are fully online, or 503 Service Unavailable on failures.
    """
    health_report = {
        "status": "healthy",
        "database": "online",
        "redis": "online"
    }
    
    # 1. Test PostgreSQL DB connectivity
    db_healthy = False
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        db_healthy = True
    except Exception as db_err:
        logger.error(f"Health check failed for Database: {db_err}")
        health_report["database"] = f"offline: {str(db_err)}"
    finally:
        db.close()

    # 2. Test Redis cache connectivity
    redis_healthy = False
    try:
        r = redis.Redis.from_url(settings.REDIS_URL)
        r.ping()
        redis_healthy = True
    except Exception as redis_err:
        logger.error(f"Health check failed for Redis: {redis_err}")
        health_report["redis"] = f"offline: {str(redis_err)}"

    # Set response status code based on aggregate health
    if not db_healthy or not redis_healthy:
        health_report["status"] = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        response.status_code = status.HTTP_200_OK

    return health_report
