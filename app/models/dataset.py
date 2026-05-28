import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    dataset_type: Mapped[str] = mapped_column(String(50), nullable=False) # e.g. CSV, IMAGE_ZIP, NUMPY_TENSOR
    status: Mapped[str] = mapped_column(String(50), default="PENDING_UPLOAD") # e.g. PENDING_UPLOAD, PROCESSING, READY, FAILED
    
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True) # S3 URI or local uploader disk path
    num_records: Mapped[int] = mapped_column(Integer, default=0) # row count, image count, or sample count
    schema_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True) # stores column datatypes, resolutions, shapes

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User")

    def __repr__(self) -> str:
        return f"<Dataset name={self.name} type={self.dataset_type} status={self.status}>"
