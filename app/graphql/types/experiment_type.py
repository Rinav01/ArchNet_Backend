import strawberry
import uuid
from datetime import datetime
from typing import List
from app.graphql.types.training_run_type import TrainingRunType

@strawberry.type
class ExperimentType:
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    @strawberry.field
    def training_runs(self, info) -> List[TrainingRunType]:
        db = info.context.db
        from app.models.training_run import TrainingRun
        runs = db.query(TrainingRun).filter(TrainingRun.experiment_id == self.id).all()
        return [
            TrainingRunType(
                id=run.id,
                project_id=run.project_id,
                training_job_id=run.training_job_id,
                accuracy=run.accuracy,
                loss=run.loss,
                metrics_json=run.metrics_json,
                config_json=run.config_json,
                created_at=run.created_at,
                updated_at=run.updated_at
            ) for run in runs
        ]
