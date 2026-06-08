import strawberry
import uuid
from datetime import datetime

@strawberry.type
class DatasetVersionType:
    id: uuid.UUID
    dataset_id: uuid.UUID
    version_number: str
    storage_path: str
    row_count: int
    column_count: int
    metadata_json: strawberry.scalars.JSON | None
    created_at: datetime
