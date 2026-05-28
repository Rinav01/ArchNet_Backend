import strawberry
import uuid
from datetime import datetime

@strawberry.type
class UserType:
    id: uuid.UUID
    email: str
    username: str
    preferences: strawberry.scalars.JSON
    role: str
    created_at: datetime
    updated_at: datetime
