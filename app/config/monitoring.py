import os
import logging
from fastapi import FastAPI

logger = logging.getLogger("mlbuilder.monitoring")

# OpenTelemetry Imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter, BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenTelemetry libraries not fully imported: {e}. Tracing will be disabled.")
    OTEL_AVAILABLE = False

# Prometheus Imports
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROM_AVAILABLE = True
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not imported. Metrics will be disabled.")
    PROM_AVAILABLE = False


def setup_monitoring(app: FastAPI):
    """Instruments the FastAPI application with Prometheus metrics and OpenTelemetry tracing."""
    
    # 1. Initialize Prometheus Metrics
    if PROM_AVAILABLE:
        try:
            # Instrument the app and expose the metrics endpoint at /metrics
            Instrumentator().instrument(app).expose(app, endpoint="/metrics")
            logger.info("Prometheus metrics successfully initialized on endpoint '/metrics'")
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}")
    
    # 2. Initialize OpenTelemetry Tracing
    if OTEL_AVAILABLE:
        try:
            # Set up resource attributes (Service Name)
            resource = Resource.create(attributes={
                ResourceAttributes.SERVICE_NAME: "mlbuilder-backend"
            })
            
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            
            # Configure OTLP Exporter pointing to Jaeger/Collector
            # Use environment variable endpoint or default to local grpc port
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
            
            try:
                # Add OTLP span processor
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, timeout=2)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info(f"OpenTelemetry initialized with OTLPSpanExporter to endpoint: {otlp_endpoint}")
            except Exception as otlp_err:
                # Fall back gracefully to console exporter if Jaeger OTLP setup crashes or endpoint is unreachable
                logger.warning(f"OTLPSpanExporter initialization failed: {otlp_err}. Falling back to ConsoleSpanExporter.")
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(SimpleSpanProcessor(console_exporter))
                
            # Instrument FastAPI application
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI application instrumented with OpenTelemetry")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
    else:
        logger.info("Observability tracing skipped (OTel not available).")
