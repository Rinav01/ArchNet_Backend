import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.model_version import ModelVersion
from sqlalchemy import String, ForeignKey, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class RegisteredModel(Base):
    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project")
    versions: Mapped[List["ModelVersion"]] = relationship(
        "ModelVersion", back_populates="registered_model", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<RegisteredModel name={self.name} project_id={self.project_id}>"
