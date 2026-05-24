import strawberry
import uuid
from datetime import datetime

@strawberry.type
class NodeType:
    id: uuid.UUID
    project_id: uuid.UUID
    type: str
    label: str
    position_x: float
    position_y: float
    config: strawberry.scalars.JSON
    input_shape: strawberry.scalars.JSON | None
    output_shape: strawberry.scalars.JSON | None
    created_at: datetime
    updated_at: datetime

@strawberry.input
class PositionInput:
    x: float
    y: float

@strawberry.input
class NodeConfigInput:
    # Use strawberry JSON scalar for arbitrary configuration parameters
    parameters: strawberry.scalars.JSON
