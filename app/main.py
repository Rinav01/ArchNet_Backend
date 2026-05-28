from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from app.config.settings import settings
from app.config.database import engine, Base
from app.auth.dependencies import get_graphql_context
from app.graphql.schema import schema
from app.graphql.ws_router import ws_router

from app.config.logging import setup_structured_logging
from app.config.middleware import RequestTracingMiddleware
from app.config.health import health_router

# Initialize structured JSON logging
setup_structured_logging()

app = FastAPI(
    title="MLBuilder Backend",
    description="Production-Ready MLBuilder Backend API",
    version="1.0.0"
)

# Create database tables at application startup (MVP local convenience)
@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)


# Set up CORS and Request Tracing middlewares
app.add_middleware(RequestTracingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Strawberry GraphQL FastAPI router with custom context dependency
graphql_router = GraphQLRouter(
    schema,
    context_getter=get_graphql_context
)

# Include the GraphQL endpoint
app.include_router(graphql_router, prefix="/graphql")

# Include the WebSocket endpoint
app.include_router(ws_router)

# Include deep health check endpoint
app.include_router(health_router)

@app.get("/")
def read_root():
    """Service health check endpoint"""
    return {
        "status": "healthy",
        "service": "MLBuilder Backend API",
        "graphql_endpoint": "/graphql"
    }

# Local Storage Fallback Mock Uploader Endpoint
import os
import shutil
from fastapi import File, UploadFile

@app.post("/api/storage/upload/{dataset_id}")
async def mock_upload_file(dataset_id: str, filename: str, file: UploadFile = File(...)):
    """FastAPI POST endpoint receiving multipart file streams offline
    and writing them directly to local workspace scratch folders.
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

# GCP Vertex AI Webhook Callback Endpoint
import json
import uuid
from app.config.database import SessionLocal
from app.models.training_job import TrainingJob
from app.services.event_dispatcher import EventDispatcher

@app.post("/api/training/gcp-webhook")
async def gcp_training_webhook(payload: dict):
    """Secure endpoint receiving webhook callbacks from active serverless GPU training containers
    in Google Cloud Vertex AI to publish metrics in real time over WebSockets.
    """
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

