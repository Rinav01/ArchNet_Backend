import strawberry
import uuid
from datetime import datetime

@strawberry.type
class TrainingRunType:
    id: uuid.UUID
    project_id: uuid.UUID
    training_job_id: uuid.UUID | None
    
    accuracy: float
    loss: float
    
    metrics_json: strawberry.scalars.JSON | None
    config_json: strawberry.scalars.JSON | None
    
    created_at: datetime
    updated_at: datetime
