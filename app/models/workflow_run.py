import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.workflow import Workflow
from sqlalchemy import String, ForeignKey, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False) # e.g. PENDING, RUNNING, COMPLETED, FAILED
    trigger_event: Mapped[str] = mapped_column(String(100), nullable=False)
    triggered_by_resource_id: Mapped[str | None] = mapped_column(String(255), nullable=True) # UUID or identification string
    execution_logs: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    workflow: Mapped["Workflow"] = relationship("Workflow", back_populates="runs")

    def __repr__(self) -> str:
        return f"<WorkflowRun id={self.id} status={self.status}>"
