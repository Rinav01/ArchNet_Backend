import uuid
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models.registered_model import RegisteredModel
from app.models.model_version import ModelVersion
from app.models.model_artifact import ModelArtifact
from app.services.s3_service import S3Service

class ModelRegistryService:
    @staticmethod
    def register_model(db: Session, project_id: uuid.UUID, name: str, description: str = None) -> RegisteredModel:
        """Registers a new named model in the project. Model names are unique globally."""
        existing = db.query(RegisteredModel).filter(RegisteredModel.name == name.strip()).first()
        if existing:
            raise ValueError(f"Model with name '{name}' already registered.")

        model = RegisteredModel(
            project_id=project_id,
            name=name.strip(),
            description=description
        )
        db.add(model)
        db.commit()
        db.refresh(model)

        # Trigger workflow automation
        try:
            from app.services.workflow_service import WorkflowService
            WorkflowService.trigger_workflows_for_event(
                db,
                event_type="MODEL_REGISTERED",
                resource_id=model.id,
                project_id=model.project_id
            )
        except Exception as w_err:
            print(f"[Workflow Triggers Warning] failed: {w_err}")

        return model

    @staticmethod
    def create_version(
        db: Session,
        model_id: uuid.UUID,
        version: str,
        description: str = None,
        artifact_id: uuid.UUID = None,
        metrics: dict = None,
        config: dict = None,
        compiler_output: str = None
    ) -> ModelVersion:
        """Registers a new model version associated with a ModelArtifact."""
        # Check if version already exists for this model
        existing = db.query(ModelVersion).filter(
            ModelVersion.model_id == model_id,
            ModelVersion.version == version.strip()
        ).first()
        if existing:
            raise ValueError(f"Version '{version}' already exists for this model.")

        # If artifact_id is provided, verify it exists
        if artifact_id:
            artifact = db.query(ModelArtifact).filter(ModelArtifact.id == artifact_id).first()
            if not artifact:
                raise ValueError(f"ModelArtifact with ID {artifact_id} not found.")

        mv = ModelVersion(
            model_id=model_id,
            version=version.strip(),
            description=description,
            model_artifact_id=artifact_id,
            metrics=metrics or {},
            config=config or {},
            compiler_output=compiler_output,
            status="staging"
        )
        db.add(mv)
        db.commit()
        db.refresh(mv)

        # Trigger workflow automation
        try:
            from app.services.workflow_service import WorkflowService
            from app.models.registered_model import RegisteredModel
            reg_model = db.query(RegisteredModel).filter(RegisteredModel.id == mv.model_id).first()
            if reg_model:
                WorkflowService.trigger_workflows_for_event(
                    db,
                    event_type="MODEL_REGISTERED",
                    resource_id=mv.id,
                    project_id=reg_model.project_id
                )
        except Exception as w_err:
            print(f"[Workflow Triggers Warning] failed: {w_err}")

        return mv

    @staticmethod
    def get_model(db: Session, model_id: uuid.UUID) -> RegisteredModel | None:
        """Retrieves a registered model by its ID."""
        return db.query(RegisteredModel).filter(RegisteredModel.id == model_id).first()

    @staticmethod
    def list_versions(db: Session, model_id: uuid.UUID) -> List[ModelVersion]:
        """Lists all versions registered under the specified model ID."""
        return db.query(ModelVersion).filter(ModelVersion.model_id == model_id).order_by(ModelVersion.created_at.desc()).all()

    @staticmethod
    def promote_version(db: Session, version_id: uuid.UUID, status: str) -> ModelVersion:
        """Promotes or updates the deployment state status of a model version (e.g. staging -> production)."""
        mv = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not mv:
            raise ValueError(f"ModelVersion with ID {version_id} not found.")

        mv.status = status.strip()
        db.commit()
        db.refresh(mv)
        return mv

    @staticmethod
    def download_artifact(db: Session, version_id: uuid.UUID) -> str:
        """Generates a secure pre-signed GET download URL to retrieve the model weights."""
        mv = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not mv:
            raise ValueError(f"ModelVersion with ID {version_id} not found.")

        if not mv.model_artifact_id:
            # Fallback mockup url for testing
            return "http://localhost:8000/exports/weights_mock.pt"

        artifact = db.query(ModelArtifact).filter(ModelArtifact.id == mv.model_artifact_id).first()
        if not artifact or not artifact.artifact_path:
            return "http://localhost:8000/exports/weights_mock.pt"

        # Generate pre-signed URL using our S3Service helper
        return S3Service.generate_presigned_download_url(artifact.artifact_path)
