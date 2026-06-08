import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.user import User
from sqlalchemy import String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True)
    
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    dataset_type: Mapped[str] = mapped_column(String(50), nullable=False) # e.g. CSV, IMAGE_ZIP, NUMPY_TENSOR
    status: Mapped[str] = mapped_column(String(50), default="PENDING_UPLOAD") # e.g. PENDING_UPLOAD, PROCESSING, READY, FAILED
    
    storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True) # S3 URI or local uploader disk path
    row_count: Mapped[int | None] = mapped_column(Integer, default=0) # row count or sample count
    column_count: Mapped[int | None] = mapped_column(Integer, default=0) # column count or features count
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True) # stores column datatypes, resolutions, shapes

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User")

    # Backward compatibility properties
    @property
    def file_path(self) -> str | None:
        return self.storage_path

    @file_path.setter
    def file_path(self, val: str | None) -> None:
        self.storage_path = val

    @property
    def num_records(self) -> int:
        return self.row_count or 0

    @num_records.setter
    def num_records(self, val: int) -> None:
        self.row_count = val

    @property
    def schema_metadata(self) -> dict | None:
        return self.metadata_json

    @schema_metadata.setter
    def schema_metadata(self, val: dict | None) -> None:
        self.metadata_json = val

    def __repr__(self) -> str:
        return f"<Dataset name={self.name} type={self.dataset_type} status={self.status}>"
