import strawberry
import uuid
from datetime import datetime

@strawberry.type
class EdgeType:
    id: uuid.UUID
    project_id: uuid.UUID
    from_node_id: uuid.UUID
    to_node_id: uuid.UUID
    input_shape: strawberry.scalars.JSON | None
    output_shape: strawberry.scalars.JSON | None
    created_at: datetime
