import time
import redis
import logging
from fastapi import Request, HTTPException, status
from app.config.settings import settings

logger = logging.getLogger("mlbuilder.rate_limiter")

class RedisRateLimiter:
    _redis_client = None

    @classmethod
    def get_redis(cls):
        if cls._redis_client is None:
            cls._redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        return cls._redis_client

    @classmethod
    def is_rate_limited(cls, identifier: str, limit: int, period: int) -> bool:
        """Determines if a client has exceeded their rate limit using a sliding window log in Redis."""
        try:
            r = cls.get_redis()
            key = f"ratelimit:{identifier}"
            now = time.time()
            cutoff = now - period

            # Transactional sliding window pipeline
            pipe = r.pipeline()
            # Remove old timestamps outside of the window
            pipe.zremrangebyscore(key, 0, cutoff)
            # Count the remaining calls in the window
            pipe.zcard(key)
            # Add the current call timestamp
            pipe.zadd(key, {str(now): now})
            # Ensure the key auto-expires when the window passes
            pipe.expire(key, period)
            
            _, current_calls, _, _ = pipe.execute()
            
            if current_calls > limit:
                logger.warning(f"Rate limit exceeded for identifier '{identifier}' ({current_calls}/{limit} calls)")
                return True
            return False
        except Exception as e:
            # Fallback gracefully if Redis is unavailable
            logger.error(f"Rate limiter Redis connection error: {e}")
            return False

class RateLimitGuard:
    """FastAPI route dependency enforcing sliding-window rate limit checks."""
    def __init__(self, limit: int = 60, period: int = 60):
        self.limit = limit
        self.period = period

    def __call__(self, request: Request):
        # Resolve client identifier: prioritize authenticated user or fallback to client IP
        identifier = "anonymous"
        if hasattr(request.state, "user_id") and request.state.user_id:
            identifier = f"user:{request.state.user_id}"
        elif request.client and request.client.host:
            identifier = f"ip:{request.client.host}"
            
        if RedisRateLimiter.is_rate_limited(identifier, self.limit, self.period):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please slow down and try again later."
            )
