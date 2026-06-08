import uuid
from sqlalchemy.orm import Session
from app.models.deployment import Deployment
from app.models.model_artifact import ModelArtifact
from app.models.training_run import TrainingRun
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion

class LineageService:
    @staticmethod
    def get_lineage(db: Session, deployment_id: uuid.UUID) -> dict:
        """
        Traces the lineage of a deployment back to the model artifact, training run,
        dataset, and dataset version.
        """
        deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
        if not deployment:
            raise ValueError("Deployment not found.")

        artifact = db.query(ModelArtifact).filter(ModelArtifact.id == deployment.model_artifact_id).first()
        if not artifact:
            raise ValueError("Model artifact not found.")

        run = db.query(TrainingRun).filter(TrainingRun.id == artifact.training_run_id).first()
        
        run_data = None
        dataset_data = None
        version_data = None
        
        if run:
            run_data = run
            # Resolve dataset version and dataset
            if run.dataset_version_id:
                version_data = db.query(DatasetVersion).filter(DatasetVersion.id == run.dataset_version_id).first()
            if run.dataset_id:
                dataset_data = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
            elif version_data:
                dataset_data = db.query(Dataset).filter(Dataset.id == version_data.dataset_id).first()

        return {
            "deployment": deployment,
            "model_artifact": artifact,
            "training_run": run_data,
            "dataset": dataset_data,
            "dataset_version": version_data
        }
