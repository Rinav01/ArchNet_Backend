import strawberry
import uuid
from datetime import datetime

@strawberry.type
class WorkflowType:
    id: uuid.UUID
    project_id: uuid.UUID | None
    name: str
    trigger_event: str
    action_type: str
    config: strawberry.scalars.JSON
    is_active: bool
    created_at: datetime
    updated_at: datetime
