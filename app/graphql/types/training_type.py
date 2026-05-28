import strawberry
import uuid
from datetime import datetime
from typing import List

@strawberry.type
class TrainingJobType:
    id: uuid.UUID
    project_id: uuid.UUID
    dataset_id: uuid.UUID | None
    status: str
    epochs: int
    current_epoch: int
    loss_history: List[float] | None
    accuracy_history: List[float] | None
    metrics_metadata: strawberry.scalars.JSON | None
    created_at: datetime
    updated_at: datetime

@strawberry.type
class AutoMLRecommendationType:
    severity: str
    bottleneck: str
    recommended_action: str
