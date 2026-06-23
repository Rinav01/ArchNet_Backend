from slowapi import Limiter
from slowapi.util import get_remote_address

# Centralized SlowAPI rate limiter using client IP address as the key.
# Apply to individual routes with the @limiter.limit("N/period") decorator,
# or use the middleware in main.py to apply a global limit to all endpoints.
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
