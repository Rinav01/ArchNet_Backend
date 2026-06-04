import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from app.config.logging import trace_id_var, request_logger

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
        
        request_logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            trace_id=trace_id
        )
        
        try:
            response: Response = await call_next(request)
            
            duration = round(time.perf_counter() - start_time, 4)
            request_logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_seconds=duration,
                trace_id=trace_id
            )
            
            # Append trace ID to response headers
            response.headers["X-Trace-ID"] = trace_id
            return response
            
        except Exception as e:
            duration = round(time.perf_counter() - start_time, 4)
            request_logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                duration_seconds=duration,
                trace_id=trace_id,
                error=str(e)
            )
            raise e
        finally:
            # Clean up token context on thread exit
            trace_id_var.reset(token)
