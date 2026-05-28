import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from app.config.logging import trace_id_var

logger = logging.getLogger("mlbuilder.api")

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware intercepting requests to inject UUID correlation Trace IDs
    and log request lifecycle metadata in structured JSON.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        # Retrieve incoming client trace ID or generate a new one
        trace_header = request.headers.get("X-Trace-ID")
        trace_id = trace_header if trace_header else str(uuid.uuid4())
        
        # Bind the trace ID to the async context variable
        token = trace_id_var.set(trace_id)
        
        start_time = time.perf_counter()
        
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={"method": request.method, "path": request.url.path}
        )
        
        try:
            response: Response = await call_next(request)
            
            duration = round(time.perf_counter() - start_time, 4)
            logger.info(
                f"Request completed: {request.method} {request.url.path} - Status {response.status_code} in {duration}s",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_seconds": duration
                }
            )
            
            # Append trace ID to response headers
            response.headers["X-Trace-ID"] = trace_id
            return response
            
        except Exception as e:
            duration = round(time.perf_counter() - start_time, 4)
            logger.error(
                f"Request failed: {request.method} {request.url.path} - Details: {str(e)}",
                exc_info=True,
                extra={"method": request.method, "path": request.url.path, "duration_seconds": duration}
            )
            raise e
        finally:
            # Clean up token context on thread exit
            trace_id_var.reset(token)
