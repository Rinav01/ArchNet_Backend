import logging
import json
import time
from contextvars import ContextVar
from typing import Any

# Async/Thread-local ContextVar tracking trace ID across request cycles
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

class JSONFormatter(logging.Formatter):
    """Custom logging formatter that renders records into single-line structured JSON objects."""
    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
            "trace_id": trace_id_var.get()
        }
        
        # Include any extra custom dictionary parameters passed to log calls
        standard_attrs = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module", "msecs",
            "msg", "name", "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "message", "level"
        }
        for k, v in record.__dict__.items():
            if k not in standard_attrs:
                log_payload[k] = v

        if hasattr(record, "extra") and isinstance(record.extra, dict):
            for k, v in record.extra.items():
                log_payload[k] = v
                
        # Append exception traceback if present
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_payload)

def setup_structured_logging(level: int = logging.INFO):
    """Overrides system root logging handlers to stream structured JSON formatting."""
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        
    console_handler = logging.StreamHandler()
    formatter = JSONFormatter()
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)

    # Prevent duplicating logs in sub-loggers
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "sqlalchemy.engine"):
        l = logging.getLogger(logger_name)
        l.handlers = []
        l.propagate = True


import structlog

# Setup structlog configuration
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.render_to_log_kwargs,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Export four dedicated loggers
request_logger = structlog.get_logger("mlbuilder.request")
compiler_logger = structlog.get_logger("mlbuilder.compiler")
training_logger = structlog.get_logger("mlbuilder.training")
benchmark_logger = structlog.get_logger("mlbuilder.benchmark")
