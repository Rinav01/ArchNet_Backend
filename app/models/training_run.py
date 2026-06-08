import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.training_job import TrainingJob
    from app.models.experiment import Experiment
    from app.models.dataset import Dataset
    from app.models.dataset_version import DatasetVersion
from sqlalchemy import String, Float, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    training_job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("training_jobs.id", ondelete="SET NULL"), nullable=True)
    
    accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    loss: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    
    metrics_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    config_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    experiment_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True, index=True)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True)
    dataset_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True, index=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project: Mapped["Project"] = relationship("Project")
    training_job: Mapped["TrainingJob"] = relationship("TrainingJob")
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="training_runs")
    dataset: Mapped["Dataset"] = relationship("Dataset")
    dataset_version: Mapped["DatasetVersion"] = relationship("DatasetVersion")

    def __repr__(self) -> str:
        return f"<TrainingRun id={self.id} project_id={self.project_id} accuracy={self.accuracy:.4f} loss={self.loss:.4f}>"
