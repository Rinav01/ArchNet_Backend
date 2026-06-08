import strawberry
import uuid
from datetime import datetime

@strawberry.type
class ExportArtifactType:
    id: uuid.UUID
    project_id: uuid.UUID
    framework: str
    artifact_path: str
    checksum: str
    created_at: datetime
