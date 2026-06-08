import strawberry
import uuid
from datetime import datetime

@strawberry.type
class ModelArtifactType:
    id: uuid.UUID
    project_id: uuid.UUID
    training_run_id: uuid.UUID
    framework: str
    artifact_type: str
    artifact_path: str
    checksum: str
    version: str
    created_at: datetime
