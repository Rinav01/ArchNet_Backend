import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.model_artifact import ModelArtifact
from sqlalchemy import String, ForeignKey, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target: Mapped[str] = mapped_column(String(50), nullable=False)  # Local Endpoint, Docker, Vertex Endpoint
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # PENDING, ACTIVE, INACTIVE, FAILED
    endpoint_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project: Mapped["Project"] = relationship("Project")
    model_artifact: Mapped["ModelArtifact"] = relationship("ModelArtifact")

    def __repr__(self) -> str:
        return f"<Deployment id={self.id} project_id={self.project_id} target={self.target} status={self.status}>"
