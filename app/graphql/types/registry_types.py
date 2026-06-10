import strawberry
import uuid
from datetime import datetime
from typing import List

@strawberry.type
class ModelVersionType:
    id: uuid.UUID
    model_id: uuid.UUID
    version: str
    description: str | None
    status: str
    model_artifact_id: uuid.UUID | None
    metrics: strawberry.scalars.JSON | None
    config: strawberry.scalars.JSON | None
    compiler_output: str | None
    created_at: datetime
    updated_at: datetime

@strawberry.type
class RegisteredModelType:
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    @strawberry.field
    def versions(self, info) -> List[ModelVersionType]:
        db = info.context.db
        from app.models.model_version import ModelVersion
        db_versions = db.query(ModelVersion).filter(ModelVersion.model_id == self.id).order_by(ModelVersion.created_at.desc()).all()
        return [
            ModelVersionType(
                id=v.id,
                model_id=v.model_id,
                version=v.version,
                description=v.description,
                status=v.status,
                model_artifact_id=v.model_artifact_id,
                metrics=v.metrics,
                config=v.config,
                compiler_output=v.compiler_output,
                created_at=v.created_at,
                updated_at=v.updated_at
            ) for v in db_versions
        ]
