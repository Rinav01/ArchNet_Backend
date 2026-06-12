import strawberry
import uuid
from datetime import datetime

@strawberry.type
class WorkflowRunType:
    id: uuid.UUID
    workflow_id: uuid.UUID
    status: str
    trigger_event: str
    triggered_by_resource_id: str | None
    execution_logs: str | None
    created_at: datetime
    updated_at: datetime
