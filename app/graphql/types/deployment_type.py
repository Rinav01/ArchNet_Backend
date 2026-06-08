import strawberry
import uuid
from datetime import datetime

@strawberry.type
class DeploymentType:
    id: uuid.UUID
    project_id: uuid.UUID
    model_artifact_id: uuid.UUID
    target: str
    status: str
    endpoint_url: str | None
    created_at: datetime
    updated_at: datetime
