import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.training_run import TrainingRun
from sqlalchemy import String, ForeignKey, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    training_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    framework: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., PyTorch Weights, TensorFlow SavedModel, JAX Checkpoint, ONNX Model
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    project: Mapped["Project"] = relationship("Project")
    training_run: Mapped["TrainingRun"] = relationship("TrainingRun")

    def __repr__(self) -> str:
        return f"<ModelArtifact id={self.id} project_id={self.project_id} type={self.artifact_type} path={self.artifact_path}>"
