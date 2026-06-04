import strawberry
import uuid
from datetime import datetime

@strawberry.type
class DatasetType:
    id: uuid.UUID
    user_id: uuid.UUID
    project_id: uuid.UUID | None
    name: str
    description: strawberry.Private[str | None] = None
    dataset_type: str
    status: str
    file_path: str | None
    num_records: int
    schema_metadata: strawberry.scalars.JSON | None
    storage_path: str | None
    row_count: int | None
    column_count: int | None
    metadata_json: strawberry.scalars.JSON | None
    created_at: datetime
    updated_at: datetime

    @strawberry.field(name="description")
    def description_field(self) -> str | None:
        return self.description


@strawberry.type
class DatasetUploadPayload:
    dataset: DatasetType
    upload_url: str
