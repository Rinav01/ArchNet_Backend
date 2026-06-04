import uuid
from datetime import datetime
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
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project: Mapped["Project"] = relationship("Project")
    training_job: Mapped["TrainingJob"] = relationship("TrainingJob")

    def __repr__(self) -> str:
        return f"<TrainingRun id={self.id} project_id={self.project_id} accuracy={self.accuracy:.4f} loss={self.loss:.4f}>"
