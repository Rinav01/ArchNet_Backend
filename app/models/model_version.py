import uuid
from datetime import datetime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.models.registered_model import RegisteredModel
    from app.models.model_artifact import ModelArtifact
from sqlalchemy import String, ForeignKey, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., 'v1.0.0'
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="staging")  # e.g., 'staging', 'production', 'archived'
    
    model_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    
    # Store key metrics (accuracy, loss) and model configuration parameters
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=dict)
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=dict)
    compiler_output: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    registered_model: Mapped["RegisteredModel"] = relationship("RegisteredModel", back_populates="versions")
    model_artifact: Mapped["ModelArtifact"] = relationship("ModelArtifact")

    def __repr__(self) -> str:
        return f"<ModelVersion model_id={self.model_id} version={self.version} status={self.status}>"
