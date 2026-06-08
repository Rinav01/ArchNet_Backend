import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.deployment import Deployment
from sqlalchemy import Integer, Float, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class DeploymentMetrics(Base):
    __tablename__ = "deployment_metrics"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    deployment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    requests_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    memory_mb: Mapped[float] = mapped_column(Float, default=0.0)
    gpu_usage_pct: Mapped[float] = mapped_column(Float, default=0.0)

    # Relationships
    deployment: Mapped["Deployment"] = relationship("Deployment")

    def __repr__(self) -> str:
        return f"<DeploymentMetrics id={self.id} deployment_id={self.deployment_id} requests={self.requests_count} latency={self.latency_ms}>"
