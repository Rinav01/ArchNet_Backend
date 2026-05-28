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

    @strawberry.field
    def parameter_count(self) -> int:
        from app.services.memory_estimator import MemoryEstimator
        from app.models.node import Node
        # Cast self to Node model structure
        metrics = MemoryEstimator.estimate_node_metrics(self)
        return metrics["parameter_count"]

    @strawberry.field
    def parameter_memory_mb(self) -> float:
        from app.services.memory_estimator import MemoryEstimator
        metrics = MemoryEstimator.estimate_node_metrics(self)
        return metrics["parameter_memory_mb"]

    @strawberry.field
    def activation_memory_mb(self) -> float:
        from app.services.memory_estimator import MemoryEstimator
        metrics = MemoryEstimator.estimate_node_metrics(self)
        return metrics["activation_memory_mb"]

    @strawberry.field
    def flops(self) -> float:
        from app.services.memory_estimator import MemoryEstimator
        metrics = MemoryEstimator.estimate_node_metrics(self)
        return metrics["flops"]

@strawberry.input
class PositionInput:
    x: float
    y: float

@strawberry.input
class NodeConfigInput:
    # Use strawberry JSON scalar for arbitrary configuration parameters
    parameters: strawberry.scalars.JSON
