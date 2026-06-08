import strawberry
from app.graphql.types.deployment_type import DeploymentType
from app.graphql.types.model_artifact_type import ModelArtifactType
from app.graphql.types.training_run_type import TrainingRunType
from app.graphql.types.dataset_type import DatasetType
from app.graphql.types.dataset_version_type import DatasetVersionType

@strawberry.type
class LineageType:
    deployment: DeploymentType
    model_artifact: ModelArtifactType
    training_run: TrainingRunType | None
    dataset: DatasetType | None
    dataset_version: DatasetVersionType | None
