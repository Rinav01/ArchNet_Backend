import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.edge import Edge
from sqlalchemy import String, Float, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.config.database import Base

class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    type: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., 'Input', 'Conv2D', 'Dense'
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    position_x: Mapped[float] = mapped_column(Float, default=0.0)
    position_y: Mapped[float] = mapped_column(Float, default=0.0)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)  # Stores layer parameters (filters, kernel_size, activation, etc.)
    input_shape: Mapped[list | None] = mapped_column(JSONB, nullable=True)  # e.g., [None, 3, 224, 224]
    output_shape: Mapped[list | None] = mapped_column(JSONB, nullable=True) # e.g., [None, 64, 112, 112]

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="nodes")
    
    # We can also define back references for edges (from/to)
    incoming_edges: Mapped[list["Edge"]] = relationship(
        "Edge", 
        foreign_keys="[Edge.to_node_id]", 
        back_populates="to_node", 
        cascade="all, delete-orphan"
    )
    outgoing_edges: Mapped[list["Edge"]] = relationship(
        "Edge", 
        foreign_keys="[Edge.from_node_id]", 
        back_populates="from_node", 
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Node type={self.type} label={self.label}>"
