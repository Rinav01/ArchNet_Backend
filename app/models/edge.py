import uuid
from datetime import datetime
from sqlalchemy import ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class Edge(Base):
    __tablename__ = "edges"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    from_node_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False)
    to_node_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False)
    input_shape: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    output_shape: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="edges")
    from_node: Mapped["Node"] = relationship("Node", foreign_keys=[from_node_id], back_populates="outgoing_edges")
    to_node: Mapped["Node"] = relationship("Node", foreign_keys=[to_node_id], back_populates="incoming_edges")

    def __repr__(self) -> str:
        return f"<Edge from={self.from_node_id} to={self.to_node_id}>"
