import strawberry
from datetime import datetime
import uuid

@strawberry.type
class AuditLogType:
    id: strawberry.ID
    user_id: strawberry.ID | None
    action: str
    resource_type: str
    resource_id: str | None
    details: strawberry.scalars.JSON | None
    ip_address: str | None
    created_at: datetime
