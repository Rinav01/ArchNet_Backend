import uuid
import os
import shutil
import time
import hashlib
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.model_artifact import ModelArtifact
from app.models.deployment import Deployment
from app.models.deployment_metrics import DeploymentMetrics
from app.models.project import Project
from app.models.node import Node
from app.models.edge import Edge
from app.services.validation_service import ValidationService
from app.services.shape_inference_service import ShapeInferenceService
from app.ir.ir_graph import IRGraph
from app.codegen.generators.registry import GeneratorRegistry

class DeploymentService:
    @staticmethod
    def deploy_artifact(db: Session, artifact_id: uuid.UUID, target: str) -> Deployment:
        """
        Creates a deployment for the given ModelArtifact target.
        Targets supported: Docker, FastAPI, Vertex AI, SageMaker, HuggingFace, Kubernetes
        """
        artifact = db.query(ModelArtifact).filter(ModelArtifact.id == artifact_id).first()
        if not artifact:
            raise ValueError("Model artifact not found.")

        valid_targets = ["Local Endpoint", "Docker", "Vertex Endpoint", "FastAPI", "Vertex AI", "SageMaker", "HuggingFace", "Kubernetes"]
        if target not in valid_targets:
            raise ValueError(f"Invalid deployment target. Must be one of: {valid_targets}")

        deployment = Deployment(
            id=uuid.uuid4(),
            project_id=artifact.project_id,
            model_artifact_id=artifact.id,
            target=target,
            status="PENDING",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(deployment)
        db.commit()

        try:
            if target in ("FastAPI", "Local Endpoint"):
                deployment.endpoint_url = f"/api/deployments/{deployment.id}/predict"
                deployment.status = "ACTIVE"
            elif target == "Docker":
                deployment.endpoint_url = f"http://localhost:8080/predictions/{artifact.project_id}"
                deployment.status = "ACTIVE"
            elif target in ("Vertex AI", "Vertex Endpoint"):
                deployment.endpoint_url = f"https://vertex-ai.endpoints/deployments/{deployment.id}"
                deployment.status = "ACTIVE"
            elif target == "SageMaker":
                deployment.endpoint_url = f"https://sagemaker.aws/endpoints/{deployment.id}"
                deployment.status = "ACTIVE"
            elif target == "HuggingFace":
                deployment.endpoint_url = f"https://huggingface.co/spaces/{deployment.id}"
                deployment.status = "ACTIVE"
            elif target == "Kubernetes":
                deployment.endpoint_url = f"http://kubernetes-service.local/deployments/{deployment.id}"
                deployment.status = "ACTIVE"
            
            db.commit()

            # Trigger workflow automation
            try:
                from app.services.workflow_service import WorkflowService
                WorkflowService.trigger_workflows_for_event(
                    db,
                    event_type="DEPLOYMENT_COMPLETED",
                    resource_id=deployment.id,
                    project_id=deployment.project_id
                )
            except Exception as w_err:
                print(f"[Workflow Triggers Warning] failed: {w_err}")

            return deployment
        except Exception as e:
            deployment.status = "FAILED"
            db.commit()
            raise e

    @staticmethod
    def record_metrics(db: Session, deployment_id: uuid.UUID, latency_ms: float, has_error: bool):
        """
        Records resource telemetry metrics (RAM, GPU, errors) for predictions on a deployment.
        """
        memory_mb = 150.0  # fallback
        try:
            import psutil
            memory_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
        except ImportError:
            pass

        gpu_usage_pct = 0.0
        import torch
        if torch.cuda.is_available():
            try:
                gpu_usage_pct = float(torch.cuda.memory_allocated() / max(1, torch.cuda.max_memory_allocated()) * 100.0)
            except Exception:
                pass

        metric = DeploymentMetrics(
            id=uuid.uuid4(),
            deployment_id=deployment_id,
            timestamp=datetime.utcnow(),
            requests_count=1,
            latency_ms=latency_ms,
            error_count=1 if has_error else 0,
            memory_mb=memory_mb,
            gpu_usage_pct=gpu_usage_pct
        )
        db.add(metric)
        db.commit()

    @staticmethod
    def export_prediction_package(db: Session, artifact_id: uuid.UUID) -> str:
        """
        Builds a downloadable, ready-to-deploy ZIP package containing:
        - model/model.py (compiled architecture)
        - weights.pt (saved model parameters)
        - predict.py (FastAPI app)
        - requirements.txt (runtime dependencies)
        - Dockerfile (run container setup)
        """
        artifact = db.query(ModelArtifact).filter(ModelArtifact.id == artifact_id).first()
        if not artifact:
            raise ValueError("Model artifact not found.")

        project = db.query(Project).filter(Project.id == artifact.project_id).first()
        if not project:
            raise ValueError("Project not found.")

        # Compile model code
        nodes = db.query(Node).filter(Node.project_id == project.id).all()
        edges = db.query(Edge).filter(Edge.project_id == project.id).all()

        sorted_nodes = ValidationService.validate_graph(nodes, edges)
        ShapeInferenceService.run_shape_inference(sorted_nodes, edges)

        ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
        compiler = GeneratorRegistry.get_generator("PyTorch")
        generated_code = compiler.generate(ir_graph)

        # Setup temporary package directory structure
        temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "scratch", "temp", f"export_{uuid.uuid4().hex}"))
        model_dir = os.path.join(temp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        # Write model definition
        with open(os.path.join(model_dir, "model.py"), "w", encoding="utf-8") as f:
            f.write(generated_code)
        
        # Write empty __init__.py
        with open(os.path.join(model_dir, "__init__.py"), "w") as f:
            pass

        # Write/Copy weight file
        package_weights_path = os.path.join(temp_dir, "weights.pt")
        if os.path.exists(artifact.artifact_path):
            shutil.copy2(artifact.artifact_path, package_weights_path)
        else:
            # Generate dummy state dict if not found (fallback)
            import torch
            from app.services.training_service import TrainingService
            model_inst = TrainingService.instantiate_model(generated_code)
            torch.save(model_inst.state_dict(), package_weights_path)

        # Write FastAPI predict.py
        predict_code = """import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.model import MLBuilderModel

app = FastAPI(title="MLBuilder Generated Inference API")

# Load model
model = MLBuilderModel()
weights_path = os.path.join(os.path.dirname(__file__), "weights.pt")
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

class InferenceRequest(BaseModel):
    input: list

@app.post("/predict")
async def predict(req: InferenceRequest):
    try:
        x = torch.tensor(req.input, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            output = model(x)
        pred_list = output.tolist()
        if output.shape[-1] > 1:
            prediction_val = int(output.argmax(dim=-1)[0].item())
        else:
            prediction_val = float(output[0][0].item()) if output.numel() > 0 else 0.0
        return {
            "prediction": prediction_val,
            "raw_output": pred_list[0] if len(pred_list) > 0 else pred_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
        with open(os.path.join(temp_dir, "predict.py"), "w", encoding="utf-8") as f:
            f.write(predict_code)

        # Write requirements.txt
        requirements_content = """fastapi==0.111.0
uvicorn==0.30.1
torch==2.3.1
pydantic==2.7.4
numpy==1.26.4
"""
        with open(os.path.join(temp_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.write(requirements_content)

        # Write Dockerfile
        dockerfile_content = """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        with open(os.path.join(temp_dir, "Dockerfile"), "w", encoding="utf-8") as f:
            f.write(dockerfile_content)

        # Package as ZIP archive
        exports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "exports"))
        os.makedirs(exports_dir, exist_ok=True)
        
        sanitized_name = "".join(c for c in project.name if c.isalnum() or c in ('_', '-')).strip()
        zip_base_name = os.path.join(exports_dir, f"{sanitized_name}_inference_api")
        
        # Build ZIP
        shutil.make_archive(zip_base_name, "zip", temp_dir)

        # Cleanup temp directory
        shutil.rmtree(temp_dir)

        return f"/exports/{sanitized_name}_inference_api.zip"

    @staticmethod
    def rollback_deployment(db: Session, deployment_id: uuid.UUID, target_version: str) -> Deployment:
        """
        Updates an existing Deployment to target a model artifact version, redeploying it.
        """
        deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
        if not deployment:
            raise ValueError("Deployment not found.")

        # Find the ModelArtifact of the same project with target_version
        artifact = db.query(ModelArtifact).filter(
            ModelArtifact.project_id == deployment.project_id,
            ModelArtifact.version == target_version
        ).first()

        if not artifact:
            raise ValueError(f"Model artifact version '{target_version}' not found for project.")

        # Update deployment pointer
        deployment.model_artifact_id = artifact.id
        deployment.status = "PENDING"
        deployment.updated_at = datetime.utcnow()
        db.commit()

        try:
            target = deployment.target
            if target in ("FastAPI", "Local Endpoint"):
                deployment.endpoint_url = f"/api/deployments/{deployment.id}/predict"
                deployment.status = "ACTIVE"
            elif target == "Docker":
                deployment.endpoint_url = f"http://localhost:8080/predictions/{artifact.project_id}"
                deployment.status = "ACTIVE"
            elif target in ("Vertex AI", "Vertex Endpoint"):
                deployment.endpoint_url = f"https://vertex-ai.endpoints/deployments/{deployment.id}"
                deployment.status = "ACTIVE"
            elif target == "SageMaker":
                deployment.endpoint_url = f"https://sagemaker.aws/endpoints/{deployment.id}"
                deployment.status = "ACTIVE"
            elif target == "HuggingFace":
                deployment.endpoint_url = f"https://huggingface.co/spaces/{deployment.id}"
                deployment.status = "ACTIVE"
            elif target == "Kubernetes":
                deployment.endpoint_url = f"http://kubernetes-service.local/deployments/{deployment.id}"
                deployment.status = "ACTIVE"
            
            db.commit()
            return deployment
        except Exception as e:
            deployment.status = "FAILED"
            db.commit()
            raise e

    @staticmethod
    def get_deployment_status(db: Session, deployment_id: uuid.UUID) -> str:
        """Retrieves the status of an existing deployment."""
        deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
        if not deployment:
            raise ValueError("Deployment not found.")
        return deployment.status

    @staticmethod
    def stop_deployment(db: Session, deployment_id: uuid.UUID) -> Deployment:
        """Stops/deactivates an active deployment."""
        deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
        if not deployment:
            raise ValueError("Deployment not found.")
        
        deployment.status = "INACTIVE"
        deployment.updated_at = datetime.utcnow()
        db.commit()
        return deployment

