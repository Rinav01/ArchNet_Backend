import uuid
from datetime import datetime
from sqlalchemy import String, Integer, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    
    status: Mapped[str] = mapped_column(String(50), default="PENDING") # PENDING, RUNNING, COMPLETED, FAILED
    
    epochs: Mapped[int] = mapped_column(Integer, default=10)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    
    # Store list of float values
    loss_history: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    accuracy_history: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    
    # Store validation metrics, training speed, logs, and device configurations
    metrics_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project: Mapped["Project"] = relationship("Project")
    dataset: Mapped["Dataset"] = relationship("Dataset")

    def __repr__(self) -> str:
        return f"<TrainingJob id={self.id} status={self.status} epoch={self.current_epoch}/{self.epochs}>"
