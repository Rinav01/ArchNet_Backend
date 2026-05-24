import strawberry
import uuid
from datetime import datetime
from app.graphql.types.node_type import NodeType
from app.graphql.types.edge_type import EdgeType

@strawberry.type
class ProjectType:
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    description: str | None
    framework: str
    is_public: bool
    thumbnail_url: str | None
    created_at: datetime
    updated_at: datetime

    @strawberry.field
    def nodes(self, info) -> list[NodeType]:
        db = info.context.db
        from app.models.node import Node
        db_nodes = db.query(Node).filter(Node.project_id == self.id).all()
        return [
            NodeType(
                id=n.id,
                project_id=n.project_id,
                type=n.type,
                label=n.label,
                position_x=n.position_x,
                position_y=n.position_y,
                config=n.config,
                input_shape=n.input_shape,
                output_shape=n.output_shape,
                created_at=n.created_at,
                updated_at=n.updated_at
            ) for n in db_nodes
        ]

    @strawberry.field
    def edges(self, info) -> list[EdgeType]:
        db = info.context.db
        from app.models.edge import Edge
        db_edges = db.query(Edge).filter(Edge.project_id == self.id).all()
        return [
            EdgeType(
                id=e.id,
                project_id=e.project_id,
                from_node_id=e.from_node_id,
                to_node_id=e.to_node_id,
                input_shape=e.input_shape,
                output_shape=e.output_shape,
                created_at=e.created_at
            ) for e in db_edges
        ]
