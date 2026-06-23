import os
import json
import uuid
import hmac
import hashlib
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from strawberry.fastapi import GraphQLRouter

from app.config.settings import settings
from app.config.database import engine, Base
from app.auth.dependencies import get_graphql_context
from app.graphql.schema import schema
from app.graphql.ws_router import ws_router

from app.config.logging import setup_structured_logging
from app.config.middleware import RequestTracingMiddleware
from app.config.health import health_router
from app.config.rate_limit import limiter

# Initialize structured JSON logging
setup_structured_logging()


# ---------------------------------------------------------------------------
# Application lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # NOTE: Schema is managed exclusively by Alembic (run via entrypoint.sh).
    #       Do NOT call Base.metadata.create_all() here — that bypasses migrations.
    yield


app = FastAPI(
    title="MLBuilder Backend",
    description="Production-Ready MLBuilder Backend API",
    version="1.0.0",
    lifespan=lifespan,
)

# Wire up SlowAPI rate-limiter state and its 429 exception handler
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Prometheus metrics & OTel Tracing
from app.config.monitoring import setup_monitoring
setup_monitoring(app)

# ---------------------------------------------------------------------------
# Middleware stack
# ---------------------------------------------------------------------------
app.add_middleware(RequestTracingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GraphQL & WebSocket routers
# ---------------------------------------------------------------------------
graphql_router = GraphQLRouter(
    schema,
    context_getter=get_graphql_context
)

app.include_router(graphql_router, prefix="/graphql")
app.include_router(ws_router)
app.include_router(health_router)


# ---------------------------------------------------------------------------
# Root health ping
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    """Service health check endpoint"""
    return {
        "status": "healthy",
        "service": "MLBuilder Backend API",
        "graphql_endpoint": "/graphql"
    }


# ---------------------------------------------------------------------------
# Static exports
# ---------------------------------------------------------------------------
exports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exports"))
os.makedirs(exports_dir, exist_ok=True)
app.mount("/exports", StaticFiles(directory=exports_dir), name="exports")


# ---------------------------------------------------------------------------
# Local Storage Mock Upload Endpoints (development only)
# These endpoints write files to the local filesystem and are NOT safe or
# meaningful in a stateless cloud deployment. They are conditionally registered
# only when ENVIRONMENT == "development".
# ---------------------------------------------------------------------------
if settings.ENVIRONMENT == "development":

    @app.post("/api/storage/upload/{dataset_id}")
    async def mock_upload_file(dataset_id: str, filename: str, file: UploadFile = File(...)):
        """FastAPI POST endpoint receiving multipart file streams offline
        and writing them directly to local workspace scratch folders.
        Development only — not available in production.
        """
        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        storage_dir = os.path.join(workspace_dir, "scratch", "storage", dataset_id)
        os.makedirs(storage_dir, exist_ok=True)

        file_path = os.path.join(storage_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "local_path": file_path,
            "filename": filename
        }

    @app.put("/api/storage/upload/{dataset_id}")
    async def mock_upload_file_put(dataset_id: str, filename: str, request: Request):
        """FastAPI PUT endpoint receiving raw binary streams offline (e.g. from frontend client)
        and writing them directly to local workspace scratch folders.
        Development only — not available in production.
        """
        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        storage_dir = os.path.join(workspace_dir, "scratch", "storage", dataset_id)
        os.makedirs(storage_dir, exist_ok=True)

        file_path = os.path.join(storage_dir, filename)
        with open(file_path, "wb") as buffer:
            async for chunk in request.stream():
                buffer.write(chunk)

        return {
            "status": "success",
            "local_path": file_path,
            "filename": filename
        }


# ---------------------------------------------------------------------------
# GCP Vertex AI Webhook Callback Endpoint
# ---------------------------------------------------------------------------
from app.config.database import SessionLocal
from app.models.training_job import TrainingJob
from app.services.event_dispatcher import EventDispatcher

import logging
_webhook_logger = logging.getLogger("mlbuilder.webhook")


@app.post("/api/training/gcp-webhook")
async def gcp_training_webhook(
    payload: dict,
    x_webhook_secret: str | None = Header(default=None),
):
    """Secure endpoint receiving webhook callbacks from active serverless GPU training
    containers in Google Cloud Vertex AI to publish metrics in real time over WebSockets.

    All non-development requests must supply a valid X-Webhook-Secret header whose value
    matches the WEBHOOK_SECRET environment variable (compared via constant-time digest to
    prevent timing attacks).
    """
    # --- Authentication: HMAC shared-secret verification ---
    if settings.ENVIRONMENT != "development":
        if not x_webhook_secret or not hmac.compare_digest(
            x_webhook_secret, settings.WEBHOOK_SECRET
        ):
            _webhook_logger.warning(
                "Rejected webhook request with invalid or missing X-Webhook-Secret."
            )
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    job_id_str = payload.get("training_job_id")
    if not job_id_str:
        return {"status": "error", "message": "Missing training_job_id"}

    job_uuid = uuid.UUID(job_id_str)
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_uuid).first()
        if not job:
            return {"status": "error", "message": "Job not found"}

        status = payload.get("status")
        if status:
            job.status = status

        current_epoch = payload.get("current_epoch")
        if current_epoch is not None:
            job.current_epoch = current_epoch

        loss = payload.get("loss")
        accuracy = payload.get("accuracy")

        # Append epoch histories if present
        if loss is not None:
            hist_loss = list(job.loss_history or [])
            hist_loss.append(round(loss, 4))
            job.loss_history = hist_loss

        if accuracy is not None:
            hist_acc = list(job.accuracy_history or [])
            hist_acc.append(round(accuracy, 4))
            job.accuracy_history = hist_acc

        metrics_metadata = payload.get("metrics_metadata")
        if metrics_metadata:
            job.metrics_metadata = {**(job.metrics_metadata or {}), **metrics_metadata}

        db.commit()

        # Dispatch WS progress coordinates
        EventDispatcher.get_redis().publish(
            "mlbuilder:project:training",
            json.dumps(payload)
        )
        return {"status": "success"}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()
