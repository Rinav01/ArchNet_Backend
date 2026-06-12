from datetime import datetime
import strawberry
import uuid
from typing import List

from app.graphql.types.user_type import UserType
from app.graphql.types.project_type import ProjectType
from app.graphql.types.node_type import NodeType, PositionInput
from app.graphql.types.edge_type import EdgeType
from app.graphql.types.auth_payload import AuthPayload
from app.graphql.types.dataset_type import DatasetType, DatasetUploadPayload
from app.graphql.types.training_type import TrainingJobType, AutoMLRecommendationType
from app.graphql.types.audit_log_type import AuditLogType
from app.graphql.types.compilation_result import CompilationResult
from app.graphql.types.training_run_type import TrainingRunType
from app.graphql.types.export_artifact_type import ExportArtifactType
from app.graphql.types.scoring_type import ArchitectureScoreType
from app.graphql.types.template_type import ArchitectureTemplateType
from app.graphql.types.model_artifact_type import ModelArtifactType
from app.graphql.types.deployment_type import DeploymentType
from app.graphql.types.deployment_metrics_type import DeploymentMetricsType
from app.graphql.types.experiment_type import ExperimentType
from app.graphql.types.dataset_version_type import DatasetVersionType
from app.graphql.types.lineage_type import LineageType
from app.auth.rbac import verify_role_and_ownership
from app.services.audit_service import AuditService
from app.models.audit_log import AuditLog
from app.graphql.types.refactoring_suggestion_type import RefactoringSuggestionType
from app.graphql.types.registry_types import RegisteredModelType, ModelVersionType
from app.graphql.types.intelligence_types import DatasetAnalysisReportType, ExperimentAnalysisReportType, CostEstimateType, ExplainabilityReportType
from app.graphql.types.workflow_type import WorkflowType
from app.graphql.types.workflow_run_type import WorkflowRunType
from app.graphql.types.notebook_type import NotebookCellResultType
from app.services.notebook_execution_service import NotebookExecutionService

@strawberry.type
class CompilationValidationPayload:
    success: bool
    semantic_errors: List[str]
    compatibility_errors: List[str]
    compilation_errors: List[str]
    generated_code: str
    execution_logs: str

from app.services.auth_service import AuthService
from app.services.project_service import ProjectService

@strawberry.type
class Query:
    @strawberry.field
    def me(self, info) -> UserType:
        user = info.context.current_user
        if not user:
            raise Exception("Not authenticated.")
        return UserType(
            id=user.id,
            email=user.email,
            username=user.username,
            preferences=user.preferences or {},
            role=user.role,
            created_at=user.created_at,
            updated_at=user.updated_at
        )

    @strawberry.field
    def project(self, info, id: strawberry.ID) -> ProjectType | None:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=id)
        db = info.context.db
        try:
            project_uuid = uuid.UUID(id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.project import Project
        project = db.query(Project).filter(Project.id == project_uuid).first()
        if not project:
            return None

        return ProjectType(
            id=project.id,
            user_id=project.user_id,
            name=project.name,
            description=project.description,
            framework=project.framework,
            is_public=project.is_public,
            thumbnail_url=project.thumbnail_url,
            created_at=project.created_at,
            updated_at=project.updated_at
        )

    @strawberry.field
    def projects(
        self, 
        info, 
        limit: int = 20, 
        offset: int = 0
    ) -> List[ProjectType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        
        from app.models.project import Project
        if user.role == "admin":
            projects = db.query(Project).limit(limit).offset(offset).all()
        else:
            projects = db.query(Project).filter(Project.user_id == user.id).limit(limit).offset(offset).all()
            
        return [
            ProjectType(
                id=p.id,
                user_id=p.user_id,
                name=p.name,
                description=p.description,
                framework=p.framework,
                is_public=p.is_public,
                thumbnail_url=p.thumbnail_url,
                created_at=p.created_at,
                updated_at=p.updated_at
            ) for p in projects
        ]

    @strawberry.field
    def export_project(self, info, project_id: strawberry.ID) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")
            
        try:
            from app.models.project import Project
            project = db.query(Project).filter(Project.id == proj_uuid).first()
            target_user_id = project.user_id if user.role == "admin" else user.id
            
            from app.services.serialization_service import SerializationService
            return SerializationService.export_project(db, proj_uuid, target_user_id)
        except Exception as e:
            raise Exception(str(e))

    @strawberry.field
    def get_task_status(self, info, task_id: str) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        
        from celery.result import AsyncResult
        from app.tasks.celery_app import celery_app
        import json
        
        res = AsyncResult(task_id, app=celery_app)
        status_dict = {
            "task_id": task_id,
            "status": res.status,
            "ready": res.ready(),
        }
        if res.ready():
            if res.successful():
                status_dict["result"] = res.result
            else:
                status_dict["error"] = str(res.result)
        return json.dumps(status_dict)

    @strawberry.field
    def dataset(self, info, id: strawberry.ID) -> DatasetType | None:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], dataset_id=id)
        db = info.context.db
        try:
            dataset_uuid = uuid.UUID(id)
        except ValueError:
            raise Exception("Invalid dataset ID format.")

        from app.models.dataset import Dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
        if not dataset:
            return None

        return DatasetType(
            id=dataset.id,
            user_id=dataset.user_id,
            project_id=dataset.project_id,
            name=dataset.name,
            description=dataset.description,
            dataset_type=dataset.dataset_type,
            status=dataset.status,
            file_path=dataset.file_path,
            num_records=dataset.num_records,
            schema_metadata=dataset.schema_metadata,
            storage_path=dataset.storage_path,
            row_count=dataset.row_count,
            column_count=dataset.column_count,
            metadata_json=dataset.metadata_json,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at
        )

    @strawberry.field
    def datasets(
        self, 
        info, 
        limit: int = 20, 
        offset: int = 0
    ) -> List[DatasetType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        
        from app.models.dataset import Dataset
        if user.role == "admin":
            datasets = db.query(Dataset).limit(limit).offset(offset).all()
        else:
            datasets = db.query(Dataset).filter(Dataset.user_id == user.id).limit(limit).offset(offset).all()
            
        return [
            DatasetType(
                id=d.id,
                user_id=d.user_id,
                project_id=d.project_id,
                name=d.name,
                description=d.description,
                dataset_type=d.dataset_type,
                status=d.status,
                file_path=d.file_path,
                num_records=d.num_records,
                schema_metadata=d.schema_metadata,
                storage_path=d.storage_path,
                row_count=d.row_count,
                column_count=d.column_count,
                metadata_json=d.metadata_json,
                created_at=d.created_at,
                updated_at=d.updated_at
            ) for d in datasets
        ]

    @strawberry.field
    def get_dataset_preview(self, info, dataset_id: strawberry.ID) -> strawberry.scalars.JSON | None:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            dataset_uuid = uuid.UUID(dataset_id)
        except ValueError:
            raise Exception("Invalid dataset ID format.")

        from app.models.dataset import Dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
        if not dataset:
            raise Exception("Dataset not found.")

        if user.role != "admin" and dataset.user_id != user.id:
            raise Exception("Forbidden: You do not have permission to access this dataset.")

        if not dataset.metadata_json:
            return None

        preview = dataset.metadata_json.get("preview_data")
        if preview:
            return preview
            
        return dataset.metadata_json

    @strawberry.field
    def training_job(self, info, id: strawberry.ID) -> TrainingJobType | None:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            job_uuid = uuid.UUID(id)
        except ValueError:
            raise Exception("Invalid training job ID format.")

        from app.models.training_job import TrainingJob
        job = db.query(TrainingJob).filter(TrainingJob.id == job_uuid).first()
        if not job:
            return None

        # Confirm ownership via project
        from app.models.project import Project
        proj = db.query(Project).filter(Project.id == job.project_id).first()
        if user.role != "admin" and (not proj or proj.user_id != user.id):
            raise Exception("You do not have permission to access this training job.")

        return TrainingJobType(
            id=job.id,
            project_id=job.project_id,
            dataset_id=job.dataset_id,
            status=job.status,
            epochs=job.epochs,
            current_epoch=job.current_epoch,
            loss_history=job.loss_history,
            accuracy_history=job.accuracy_history,
            metrics_metadata=job.metrics_metadata,
            created_at=job.created_at,
            updated_at=job.updated_at
        )

    @strawberry.field
    def training_runs(self, info, project_id: strawberry.ID) -> List[TrainingRunType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.training_run import TrainingRun
        runs = db.query(TrainingRun).filter(TrainingRun.project_id == proj_uuid).order_by(TrainingRun.created_at.desc()).all()
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

    @strawberry.field
    def training_run(self, info, id: strawberry.ID) -> TrainingRunType | None:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            run_uuid = uuid.UUID(id)
        except ValueError:
            raise Exception("Invalid training run ID format.")

        from app.models.training_run import TrainingRun
        run = db.query(TrainingRun).filter(TrainingRun.id == run_uuid).first()
        if not run:
            return None

        # Verify project access
        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=run.project_id)

        return TrainingRunType(
            id=run.id,
            project_id=run.project_id,
            training_job_id=run.training_job_id,
            accuracy=run.accuracy,
            loss=run.loss,
            metrics_json=run.metrics_json,
            config_json=run.config_json,
            created_at=run.created_at,
            updated_at=run.updated_at
        )

    @strawberry.field
    def automl_suggestions(self, info, project_id: strawberry.ID) -> List[AutoMLRecommendationType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        # Check Cache
        from app.services.caching_service import CachingService
        import json
        cache_key = f"cache:project:automl:{project_id}"
        cached_res = CachingService.get(cache_key)
        if cached_res:
            try:
                suggestions_data = json.loads(cached_res)
                return [
                    AutoMLRecommendationType(
                        severity=s["severity"],
                        bottleneck=s["bottleneck"],
                        recommended_action=s["recommended_action"]
                    ) for s in suggestions_data
                ]
            except Exception:
                pass

        # Retrieve nodes and edges
        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        from app.services.automl_engine import AutoMLSuggestionEngine
        suggestions = AutoMLSuggestionEngine.analyze_architecture_bottlenecks(nodes, edges)
        
        try:
            CachingService.set(cache_key, json.dumps(suggestions), expire_seconds=3600)
        except Exception:
            pass
        
        return [
            AutoMLRecommendationType(
                severity=s["severity"],
                bottleneck=s["bottleneck"],
                recommended_action=s["recommended_action"]
            ) for s in suggestions
        ]

    @strawberry.field
    def audit_logs(self, info, limit: int = 50, offset: int = 0) -> List[AuditLogType]:
        user = verify_role_and_ownership(info, ["admin"])
        db = info.context.db
        from app.models.audit_log import AuditLog
        logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).offset(offset).all()
        return [
            AuditLogType(
                id=log.id,
                user_id=log.user_id,
                action=log.action,
                resource_type=log.resource_type,
                resource_id=log.resource_id,
                details=log.details,
                ip_address=log.ip_address,
                created_at=log.created_at
            ) for log in logs
        ]

    @strawberry.field
    def score_architecture(self, info, project_id: strawberry.ID) -> ArchitectureScoreType:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        from app.services.architecture_scorer import ArchitectureScorer
        score_data = ArchitectureScorer.score(nodes, edges)
        return ArchitectureScoreType(
            score=score_data["score"],
            grade=score_data["grade"],
            breakdown=score_data["breakdown"]
        )

    @strawberry.field
    def recommend_architecture(self, info, project_id: strawberry.ID) -> List[AutoMLRecommendationType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        from app.services.recommendation_engine import RecommendationEngine
        recommendations = RecommendationEngine.get_recommendations(nodes, edges)
        return [
            AutoMLRecommendationType(
                severity=r["severity"],
                bottleneck=r["bottleneck"],
                recommended_action=r["recommended_action"]
            ) for r in recommendations
        ]

    @strawberry.field
    def generate_architecture_template(self, info, prompt: str) -> ArchitectureTemplateType:
        user = info.context.current_user
        if not user:
            raise Exception("Not authenticated.")
        
        from app.services.template_generator import ArchitectureTemplateGenerator
        template = ArchitectureTemplateGenerator.generate(prompt)
        return ArchitectureTemplateType(
            name=template["name"],
            description=template["description"],
            nodes=template["nodes"],
            edges=template["edges"]
        )

    @strawberry.field
    def model_artifacts(self, info, project_id: strawberry.ID) -> List[ModelArtifactType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.model_artifact import ModelArtifact
        artifacts = db.query(ModelArtifact).filter(ModelArtifact.project_id == proj_uuid).all()
        return [
            ModelArtifactType(
                id=art.id,
                project_id=art.project_id,
                training_run_id=art.training_run_id,
                framework=art.framework,
                artifact_type=art.artifact_type,
                artifact_path=art.artifact_path,
                checksum=art.checksum,
                version=art.version,
                created_at=art.created_at
            ) for art in artifacts
        ]

    @strawberry.field
    def model_artifact(self, info, id: strawberry.ID) -> ModelArtifactType | None:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            art_uuid = uuid.UUID(id)
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.models.model_artifact import ModelArtifact
        art = db.query(ModelArtifact).filter(ModelArtifact.id == art_uuid).first()
        if not art:
            return None

        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(art.project_id))

        return ModelArtifactType(
            id=art.id,
            project_id=art.project_id,
            training_run_id=art.training_run_id,
            framework=art.framework,
            artifact_type=art.artifact_type,
            artifact_path=art.artifact_path,
            checksum=art.checksum,
            version=art.version,
            created_at=art.created_at
        )

    @strawberry.field
    def deployments(self, info, project_id: strawberry.ID) -> List[DeploymentType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.deployment import Deployment
        deployments = db.query(Deployment).filter(Deployment.project_id == proj_uuid).all()
        return [
            DeploymentType(
                id=d.id,
                project_id=d.project_id,
                model_artifact_id=d.model_artifact_id,
                target=d.target,
                status=d.status,
                endpoint_url=d.endpoint_url,
                created_at=d.created_at,
                updated_at=d.updated_at
            ) for d in deployments
        ]

    @strawberry.field
    def deployment(self, info, id: strawberry.ID) -> DeploymentType | None:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            dep_uuid = uuid.UUID(id)
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.models.deployment import Deployment
        d = db.query(Deployment).filter(Deployment.id == dep_uuid).first()
        if not d:
            return None

        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(d.project_id))

        return DeploymentType(
            id=d.id,
            project_id=d.project_id,
            model_artifact_id=d.model_artifact_id,
            target=d.target,
            status=d.status,
            endpoint_url=d.endpoint_url,
            created_at=d.created_at,
            updated_at=d.updated_at
        )

    @strawberry.field
    def deployment_metrics(self, info, deployment_id: strawberry.ID) -> List[DeploymentMetricsType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            dep_uuid = uuid.UUID(deployment_id)
        except ValueError:
            raise Exception("Invalid deployment ID format.")

        from app.models.deployment import Deployment
        d = db.query(Deployment).filter(Deployment.id == dep_uuid).first()
        if not d:
            raise Exception("Deployment not found.")

        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(d.project_id))

        from app.models.deployment_metrics import DeploymentMetrics
        metrics = db.query(DeploymentMetrics).filter(DeploymentMetrics.deployment_id == dep_uuid).order_by(DeploymentMetrics.timestamp.desc()).all()
        return [
            DeploymentMetricsType(
                id=m.id,
                deployment_id=m.deployment_id,
                timestamp=m.timestamp,
                requests_count=m.requests_count,
                latency_ms=m.latency_ms,
                error_count=m.error_count,
                memory_mb=m.memory_mb,
                gpu_usage_pct=m.gpu_usage_pct
            ) for m in metrics
        ]

    @strawberry.field
    def experiments(self, info, project_id: strawberry.ID) -> List[ExperimentType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.experiment import Experiment
        experiments = db.query(Experiment).filter(Experiment.project_id == proj_uuid).all()
        return [
            ExperimentType(
                id=e.id,
                project_id=e.project_id,
                name=e.name,
                description=e.description,
                created_at=e.created_at,
                updated_at=e.updated_at
            ) for e in experiments
        ]

    @strawberry.field
    def experiment(self, info, id: strawberry.ID) -> ExperimentType | None:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            exp_uuid = uuid.UUID(id)
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.models.experiment import Experiment
        e = db.query(Experiment).filter(Experiment.id == exp_uuid).first()
        if not e:
            return None

        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(e.project_id))

        return ExperimentType(
            id=e.id,
            project_id=e.project_id,
            name=e.name,
            description=e.description,
            created_at=e.created_at,
            updated_at=e.updated_at
        )

    @strawberry.field
    def dataset_versions(self, info, dataset_id: strawberry.ID) -> List[DatasetVersionType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            ds_uuid = uuid.UUID(dataset_id)
        except ValueError:
            raise Exception("Invalid dataset ID format.")

        from app.models.dataset import Dataset
        dataset = db.query(Dataset).filter(Dataset.id == ds_uuid).first()
        if not dataset:
            raise Exception("Dataset not found.")

        if user.role != "admin" and dataset.user_id != user.id:
            raise Exception("Forbidden: You do not have permission to view versions for this dataset.")

        from app.models.dataset_version import DatasetVersion
        versions = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == ds_uuid).order_by(DatasetVersion.created_at.desc()).all()
        return [
            DatasetVersionType(
                id=v.id,
                dataset_id=v.dataset_id,
                version_number=v.version_number,
                storage_path=v.storage_path,
                row_count=v.row_count,
                column_count=v.column_count,
                metadata_json=v.metadata_json,
                created_at=v.created_at
            ) for v in versions
        ]

    @strawberry.field
    def get_lineage(self, info, deployment_id: strawberry.ID) -> LineageType:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            dep_uuid = uuid.UUID(deployment_id)
        except ValueError:
            raise Exception("Invalid deployment ID format.")

        from app.services.lineage_service import LineageService
        lineage = LineageService.get_lineage(db, dep_uuid)
        
        d = lineage["deployment"]
        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(d.project_id))

        d_type = DeploymentType(
            id=d.id,
            project_id=d.project_id,
            model_artifact_id=d.model_artifact_id,
            target=d.target,
            status=d.status,
            endpoint_url=d.endpoint_url,
            created_at=d.created_at,
            updated_at=d.updated_at
        )

        art = lineage["model_artifact"]
        art_type = ModelArtifactType(
            id=art.id,
            project_id=art.project_id,
            training_run_id=art.training_run_id,
            framework=art.framework,
            artifact_type=art.artifact_type,
            artifact_path=art.artifact_path,
            checksum=art.checksum,
            version=art.version,
            created_at=art.created_at
        )

        run_type = None
        run = lineage["training_run"]
        if run:
            run_type = TrainingRunType(
                id=run.id,
                project_id=run.project_id,
                training_job_id=run.training_job_id,
                accuracy=run.accuracy,
                loss=run.loss,
                metrics_json=run.metrics_json,
                config_json=run.config_json,
                created_at=run.created_at,
                updated_at=run.updated_at
            )

        dataset_type = None
        ds = lineage["dataset"]
        if ds:
            dataset_type = DatasetType(
                id=ds.id,
                user_id=ds.user_id,
                project_id=ds.project_id,
                name=ds.name,
                description=ds.description,
                dataset_type=ds.dataset_type,
                status=ds.status,
                file_path=ds.file_path,
                num_records=ds.num_records,
                schema_metadata=ds.schema_metadata,
                storage_path=ds.storage_path,
                row_count=ds.row_count,
                column_count=ds.column_count,
                metadata_json=ds.metadata_json,
                created_at=ds.created_at,
                updated_at=ds.updated_at
            )

        ver_type = None
        ver = lineage["dataset_version"]
        if ver:
            ver_type = DatasetVersionType(
                id=ver.id,
                dataset_id=ver.dataset_id,
                version_number=ver.version_number,
                storage_path=ver.storage_path,
                row_count=ver.row_count,
                column_count=ver.column_count,
                metadata_json=ver.metadata_json,
                created_at=ver.created_at
            )

        return LineageType(
            deployment=d_type,
            model_artifact=art_type,
            training_run=run_type,
            dataset=dataset_type,
            dataset_version=ver_type
        )

    @strawberry.field
    def explain_architecture(self, info, project_id: strawberry.ID) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")
            
        try:
            from app.services.copilot.copilot_service import CopilotService
            explanation = CopilotService.explain_architecture(db, proj_uuid)
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="EXPLAIN_ARCHITECTURE",
                resource_type="PROJECT",
                resource_id=project_id,
                ip_address=info.context.ip_address
            )
            return explanation
        except Exception as e:
            raise Exception(str(e))

    @strawberry.field
    def refactor_architecture(self, info, project_id: strawberry.ID) -> List[RefactoringSuggestionType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")
            
        try:
            from app.services.copilot.copilot_service import CopilotService
            suggestions = CopilotService.refactor_architecture(db, proj_uuid)
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="REFACTOR_ARCHITECTURE",
                resource_type="PROJECT",
                resource_id=project_id,
                ip_address=info.context.ip_address
            )
            
            from app.graphql.types.refactoring_suggestion_type import RefactorActionType, RefactoringSuggestionType
            
            res = []
            for s in suggestions:
                act = s.get("action")
                act_type = None
                if act:
                    act_type = RefactorActionType(
                        type=act["type"],
                        params=act.get("params")
                    )
                res.append(RefactoringSuggestionType(
                    category=s["category"],
                    description=s["description"],
                    action=act_type
                ))
            return res
        except Exception as e:
            raise Exception(str(e))

    @strawberry.field
    def analyze_dataset(self, info, dataset_id: strawberry.ID) -> DatasetAnalysisReportType:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], dataset_id=dataset_id)
        db = info.context.db
        try:
            ds_uuid = uuid.UUID(dataset_id)
        except ValueError:
            raise Exception("Invalid dataset ID format.")

        from app.services.dataset_analysis_service import DatasetAnalysisService
        res = DatasetAnalysisService.analyze_dataset_report(db, ds_uuid)

        img_stats = None
        if res.get("image_stats"):
            img = res["image_stats"]
            from app.graphql.types.intelligence_types import DatasetImageStatsType
            img_stats = DatasetImageStatsType(
                image_count=img["image_count"],
                classes=img["classes"],
                class_counts=img["class_counts"],
                min_resolution=img["min_resolution"],
                max_resolution=img["max_resolution"],
                imbalance_ratio=img["imbalance_ratio"],
                is_imbalanced=img["is_imbalanced"]
            )

        csv_stats = None
        if res.get("csv_stats"):
            csv = res["csv_stats"]
            from app.graphql.types.intelligence_types import DatasetCSVStatsType
            csv_stats = DatasetCSVStatsType(
                missing_values=csv["missing_values"],
                outliers=csv["outliers"],
                correlations=csv["correlations"]
            )

        text_stats = None
        if res.get("text_stats"):
            txt = res["text_stats"]
            from app.graphql.types.intelligence_types import DatasetTextStatsType
            text_stats = DatasetTextStatsType(
                vocab_size=txt["vocab_size"],
                total_tokens=txt["total_tokens"],
                top_tokens=txt["top_tokens"],
                min_seq_len=txt["min_seq_len"],
                max_seq_len=txt["max_seq_len"],
                mean_seq_len=txt["mean_seq_len"]
            )

        return DatasetAnalysisReportType(
            format=res["format"],
            row_count=res["row_count"],
            column_count=res["column_count"],
            image_stats=img_stats,
            csv_stats=csv_stats,
            text_stats=text_stats,
            recommendations=res["recommendations"]
        )

    @strawberry.field
    def analyze_experiment(self, info, run_id: strawberry.ID) -> ExperimentAnalysisReportType:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            run_uuid = uuid.UUID(run_id)
        except ValueError:
            raise Exception("Invalid training run ID format.")

        from app.models.training_run import TrainingRun
        run = db.query(TrainingRun).filter(TrainingRun.id == run_uuid).first()
        if not run:
            raise Exception("Training run not found.")

        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(run.project_id))

        from app.services.experiment_analysis_service import ExperimentAnalysisService
        res = ExperimentAnalysisService.analyze_experiment_run(db, run_uuid)

        return ExperimentAnalysisReportType(
            fit_type=res["fit_type"],
            is_stable=res["is_stable"],
            loss_history=res["loss_history"],
            accuracy_history=res["accuracy_history"],
            val_loss_history=res["val_loss_history"],
            val_accuracy_history=res["val_accuracy_history"],
            recommendations=res["recommendations"]
        )

    @strawberry.field
    def registered_models(self, info, project_id: strawberry.ID) -> List[RegisteredModelType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.registered_model import RegisteredModel
        models = db.query(RegisteredModel).filter(RegisteredModel.project_id == proj_uuid).all()
        return [
            RegisteredModelType(
                id=m.id,
                project_id=m.project_id,
                name=m.name,
                description=m.description,
                created_at=m.created_at,
                updated_at=m.updated_at
            ) for m in models
        ]

    @strawberry.field
    def get_model(self, info, model_id: strawberry.ID) -> RegisteredModelType | None:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            model_uuid = uuid.UUID(model_id)
        except ValueError:
            raise Exception("Invalid model ID format.")

        from app.models.registered_model import RegisteredModel
        model = db.query(RegisteredModel).filter(RegisteredModel.id == model_uuid).first()
        if not model:
            return None

        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(model.project_id))

        return RegisteredModelType(
            id=model.id,
            project_id=model.project_id,
            name=model.name,
            description=model.description,
            created_at=model.created_at,
            updated_at=model.updated_at
        )

    @strawberry.field
    def list_versions(self, info, model_id: strawberry.ID) -> List[ModelVersionType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            model_uuid = uuid.UUID(model_id)
        except ValueError:
            raise Exception("Invalid model ID format.")

        from app.models.registered_model import RegisteredModel
        model = db.query(RegisteredModel).filter(RegisteredModel.id == model_uuid).first()
        if not model:
            raise Exception("Model not found.")

        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(model.project_id))

        from app.services.model_registry_service import ModelRegistryService
        db_versions = ModelRegistryService.list_versions(db, model_uuid)

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

    @strawberry.field
    def download_artifact(self, info, version_id: strawberry.ID) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            ver_uuid = uuid.UUID(version_id)
        except ValueError:
            raise Exception("Invalid version ID format.")

        from app.models.model_version import ModelVersion
        from app.models.registered_model import RegisteredModel
        mv = db.query(ModelVersion).filter(ModelVersion.id == ver_uuid).first()
        if not mv:
            raise Exception("Model version not found.")

        model = db.query(RegisteredModel).filter(RegisteredModel.id == mv.model_id).first()
        if not model:
            raise Exception("Registered model not found.")

        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(model.project_id))

        from app.services.model_registry_service import ModelRegistryService
        return ModelRegistryService.download_artifact(db, ver_uuid)

    @strawberry.field
    def estimate_costs(
        self,
        info,
        project_id: strawberry.ID,
        dataset_id: strawberry.ID | None = None,
        epochs: int = 10,
        gpu_type: str = "T4"
    ) -> CostEstimateType:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
            ds_uuid = uuid.UUID(dataset_id) if dataset_id else None
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.services.cost_estimator import CostEstimator
        res = CostEstimator.estimate_costs_for_project(
            db=db,
            project_id=proj_uuid,
            dataset_id=ds_uuid,
            epochs=epochs,
            gpu_type=gpu_type
        )
        return CostEstimateType(**res)

    @strawberry.field
    def explain_model(self, info, project_id: strawberry.ID) -> ExplainabilityReportType:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.services.explainability_agent import ExplainabilityAgent
        res = ExplainabilityAgent.generate_explanation(db, proj_uuid)
        return ExplainabilityReportType(**res)

    @strawberry.field
    def get_deployment_status(self, info, deployment_id: strawberry.ID) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            dep_uuid = uuid.UUID(deployment_id)
        except ValueError:
            raise Exception("Invalid deployment ID format.")

        from app.models.deployment import Deployment
        dep = db.query(Deployment).filter(Deployment.id == dep_uuid).first()
        if not dep:
            raise Exception("Deployment not found.")

        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(dep.project_id))

        from app.services.deployment_service import DeploymentService
        return DeploymentService.get_deployment_status(db, dep_uuid)

    @strawberry.field
    def workflows(self, info, project_id: strawberry.ID | None = None) -> List[WorkflowType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        
        proj_uuid = uuid.UUID(project_id) if project_id else None
        if proj_uuid:
            verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(proj_uuid))

        from app.services.workflow_service import WorkflowService
        db_workflows = WorkflowService.list_workflows(db, proj_uuid)

        return [
            WorkflowType(
                id=w.id,
                project_id=w.project_id,
                name=w.name,
                trigger_event=w.trigger_event,
                action_type=w.action_type,
                config=w.config,
                is_active=w.is_active,
                created_at=w.created_at,
                updated_at=w.updated_at
            ) for w in db_workflows
        ]

    @strawberry.field
    def workflow_runs(self, info, workflow_id: strawberry.ID) -> List[WorkflowRunType]:
        user = verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            wf_uuid = uuid.UUID(workflow_id)
        except ValueError:
            raise Exception("Invalid workflow ID format.")

        from app.services.workflow_service import WorkflowService
        wf = WorkflowService.get_workflow(db, wf_uuid)
        if not wf:
            raise Exception("Workflow not found.")

        if wf.project_id:
            verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(wf.project_id))

        from app.models.workflow_run import WorkflowRun
        db_runs = db.query(WorkflowRun).filter(WorkflowRun.workflow_id == wf_uuid).order_by(WorkflowRun.created_at.desc()).all()

        return [
            WorkflowRunType(
                id=r.id,
                workflow_id=r.workflow_id,
                status=r.status,
                trigger_event=r.trigger_event,
                triggered_by_resource_id=r.triggered_by_resource_id,
                execution_logs=r.execution_logs,
                created_at=r.created_at,
                updated_at=r.updated_at
            ) for r in db_runs
        ]

@strawberry.type
class Mutation:
    @strawberry.mutation
    def signup(
        self, 
        email: str, 
        username: str, 
        password: str,
        info
    ) -> AuthPayload:
        db = info.context.db
        try:
            user = AuthService.signup(db, email, username, password)
            token, refresh_token = AuthService.generate_tokens(db, user)
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="SIGNUP",
                resource_type="USER",
                resource_id=str(user.id),
                ip_address=info.context.ip_address
            )
            
            user_gql = UserType(
                id=user.id,
                email=user.email,
                username=user.username,
                preferences=user.preferences or {},
                role=user.role,
                created_at=user.created_at,
                updated_at=user.updated_at
            )
            return AuthPayload(token=token, refresh_token=refresh_token, user=user_gql)
        except ValueError as e:
            raise Exception(str(e))

    @strawberry.mutation
    def login(
        self, 
        email: str, 
        password: str,
        info
    ) -> AuthPayload:
        db = info.context.db
        try:
            token, refresh_token, user = AuthService.login(db, email, password)
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="LOGIN",
                resource_type="USER",
                resource_id=str(user.id),
                ip_address=info.context.ip_address
            )
            
            user_gql = UserType(
                id=user.id,
                email=user.email,
                username=user.username,
                preferences=user.preferences or {},
                role=user.role,
                created_at=user.created_at,
                updated_at=user.updated_at
            )
            return AuthPayload(token=token, refresh_token=refresh_token, user=user_gql)
        except ValueError as e:
            raise Exception(str(e))

    @strawberry.mutation
    def rotate_refresh_token(
        self,
        refresh_token: str,
        info
    ) -> AuthPayload:
        db = info.context.db
        try:
            token, new_refresh_token, user = AuthService.rotate_refresh_token(db, refresh_token)
            user_gql = UserType(
                id=user.id,
                email=user.email,
                username=user.username,
                preferences=user.preferences or {},
                role=user.role,
                created_at=user.created_at,
                updated_at=user.updated_at
            )
            return AuthPayload(token=token, refresh_token=new_refresh_token, user=user_gql)
        except ValueError as e:
            raise Exception(str(e))

    @strawberry.mutation
    def create_project(
        self, 
        name: str, 
        description: str | None = None,
        framework: str = "PyTorch",
        info = None
    ) -> ProjectType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        project = ProjectService.create_project(
            db, 
            user_id=user.id, 
            name=name, 
            description=description, 
            framework=framework
        )
        # Log Audit trail
        AuditService.log_action(
            db,
            user_id=user.id,
            action="CREATE_PROJECT",
            resource_type="PROJECT",
            resource_id=str(project.id),
            details={"name": name, "framework": framework},
            ip_address=info.context.ip_address
        )
        return ProjectType(
            id=project.id,
            user_id=project.user_id,
            name=project.name,
            description=project.description,
            framework=project.framework,
            is_public=project.is_public,
            thumbnail_url=project.thumbnail_url,
            created_at=project.created_at,
            updated_at=project.updated_at
        )

    @strawberry.mutation
    def add_node(
        self,
        project_id: strawberry.ID,
        type: str,
        label: str,
        position: PositionInput,
        config: strawberry.scalars.JSON,
        info
    ) -> NodeType:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        node = ProjectService.add_node(
            db, 
            project_id=proj_uuid, 
            node_type=type, 
            label=label, 
            position_x=position.x, 
            position_y=position.y, 
            config=config
        )
        # Log Audit trail
        AuditService.log_action(
            db,
            user_id=user.id,
            action="ADD_NODE",
            resource_type="NODE",
            resource_id=str(node.id),
            details={"project_id": project_id, "type": type, "label": label},
            ip_address=info.context.ip_address
        )
        return NodeType(
            id=node.id,
            project_id=node.project_id,
            type=node.type,
            label=node.label,
            position_x=node.position_x,
            position_y=node.position_y,
            config=node.config,
            input_shape=node.input_shape,
            output_shape=node.output_shape,
            created_at=node.created_at,
            updated_at=node.updated_at
        )

    @strawberry.mutation
    def add_edge(
        self,
        project_id: strawberry.ID,
        from_node_id: strawberry.ID,
        to_node_id: strawberry.ID,
        info
    ) -> EdgeType:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
            from_uuid = uuid.UUID(from_node_id)
            to_uuid = uuid.UUID(to_node_id)
        except ValueError:
            raise Exception("Invalid UUID format for IDs.")

        try:
            edge = ProjectService.add_edge(db, proj_uuid, from_uuid, to_uuid)
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="ADD_EDGE",
                resource_type="EDGE",
                resource_id=str(edge.id),
                details={"project_id": project_id, "from_node_id": from_node_id, "to_node_id": to_node_id},
                ip_address=info.context.ip_address
            )
            return EdgeType(
                id=edge.id,
                project_id=edge.project_id,
                from_node_id=edge.from_node_id,
                to_node_id=edge.to_node_id,
                input_shape=edge.input_shape,
                output_shape=edge.output_shape,
                created_at=edge.created_at
            )
        except ValueError as e:
            raise Exception(str(e))

    @strawberry.mutation
    def delete_node(
        self,
        project_id: strawberry.ID,
        node_id: strawberry.ID,
        info
    ) -> bool:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
            node_uuid = uuid.UUID(node_id)
        except ValueError:
            raise Exception("Invalid UUID format.")

        res = ProjectService.delete_node(db, proj_uuid, node_uuid)
        if res:
            AuditService.log_action(
                db,
                user_id=user.id,
                action="DELETE_NODE",
                resource_type="NODE",
                resource_id=node_id,
                details={"project_id": project_id},
                ip_address=info.context.ip_address
            )
        return res

    @strawberry.mutation
    def delete_edge(
        self,
        project_id: strawberry.ID,
        edge_id: strawberry.ID,
        info
    ) -> bool:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
            edge_uuid = uuid.UUID(edge_id)
        except ValueError:
            raise Exception("Invalid UUID format.")

        res = ProjectService.delete_edge(db, proj_uuid, edge_uuid)
        if res:
            AuditService.log_action(
                db,
                user_id=user.id,
                action="DELETE_EDGE",
                resource_type="EDGE",
                resource_id=edge_id,
                details={"project_id": project_id},
                ip_address=info.context.ip_address
            )
        return res

    @strawberry.mutation
    def delete_project(
        self,
        id: strawberry.ID,
        info
    ) -> bool:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        res = ProjectService.delete_project(db, proj_uuid)
        if res:
            AuditService.log_action(
                db,
                user_id=user.id,
                action="DELETE_PROJECT",
                resource_type="PROJECT",
                resource_id=id,
                ip_address=info.context.ip_address
            )
        return res

    @strawberry.mutation
    def import_project(self, info, name: str, graph_data: str) -> ProjectType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            from app.services.serialization_service import SerializationService
            project = SerializationService.import_project(db, user.id, name, graph_data)
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="IMPORT_PROJECT",
                resource_type="PROJECT",
                resource_id=str(project.id),
                details={"name": name},
                ip_address=info.context.ip_address
            )
            return ProjectType(
                id=project.id,
                user_id=project.user_id,
                name=project.name,
                description=project.description,
                framework=project.framework,
                is_public=project.is_public,
                thumbnail_url=project.thumbnail_url,
                created_at=project.created_at,
                updated_at=project.updated_at
            )
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def compile_project(
        self,
        project_id: strawberry.ID,
        info
    ) -> CompilationResult:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        # Fetch project with role-sensitive checks
        from app.models.project import Project
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)
            
        if not project:
            raise Exception("Project not found.")

        # Fetch nodes and edges
        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        try:
            # Validate and sort
            from app.services.validation_service import ValidationService
            sorted_nodes = ValidationService.validate_graph(nodes, edges)

            # Infer shapes
            from app.services.shape_inference_service import ShapeInferenceService
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
            db.commit()

            # Compile using framework-agnostic IRGraph
            from app.ir.ir_graph import IRGraph
            ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
            
            # Apply graph optimizations
            from app.services.graph_engine import GraphOptimizer
            GraphOptimizer.simplify_graph(ir_graph)

            # Generate PyTorch code
            from app.codegen.generators.registry import GeneratorRegistry
            compiler = GeneratorRegistry.get_generator("PyTorch")
            generated_code = compiler.generate(ir_graph)
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="COMPILE_PROJECT",
                resource_type="PROJECT",
                resource_id=str(project.id),
                ip_address=info.context.ip_address
            )
            return CompilationResult(code=generated_code)
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def generate_pytorch_code(
        self,
        project_id: strawberry.ID,
        info
    ) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        # Fetch project with role-sensitive checks
        from app.models.project import Project
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)
            
        if not project:
            raise Exception("Project not found.")

        # Check Cache
        from app.services.caching_service import CachingService
        cache_key = f"cache:project:pytorch:{project_id}"
        cached_code = CachingService.get(cache_key)
        if cached_code:
            return cached_code

        # Fetch nodes and edges
        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        try:
            # Validate and sort
            from app.services.validation_service import ValidationService
            sorted_nodes = ValidationService.validate_graph(nodes, edges)

            # Infer shapes
            from app.services.shape_inference_service import ShapeInferenceService
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)

            # Persist computed shapes back to database
            db.commit()

            # Compile using framework-agnostic IRGraph
            from app.ir.ir_graph import IRGraph
            ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
            
            # Apply graph optimizations
            from app.services.graph_engine import GraphOptimizer
            GraphOptimizer.simplify_graph(ir_graph)

            # Generate PyTorch code
            from app.codegen.generators.registry import GeneratorRegistry
            compiler = GeneratorRegistry.get_generator("PyTorch")
            generated_code = compiler.generate(ir_graph)
            
            # Save to Cache
            CachingService.set(cache_key, generated_code, expire_seconds=3600)
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="GENERATE_PYTORCH_CODE",
                resource_type="PROJECT",
                resource_id=str(project.id),
                ip_address=info.context.ip_address
            )
            return generated_code
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def generate_tensorflow_code(
        self,
        project_id: strawberry.ID,
        info
    ) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        # Fetch project with role-sensitive checks
        from app.models.project import Project
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)
            
        if not project:
            raise Exception("Project not found.")

        # Check Cache
        from app.services.caching_service import CachingService
        cache_key = f"cache:project:tensorflow:{project_id}"
        cached_code = CachingService.get(cache_key)
        if cached_code:
            return cached_code

        # Fetch nodes and edges
        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        try:
            # Validate and sort
            from app.services.validation_service import ValidationService
            sorted_nodes = ValidationService.validate_graph(nodes, edges)

            # Infer shapes
            from app.services.shape_inference_service import ShapeInferenceService
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)

            # Persist computed shapes back to database
            db.commit()

            # Compile using framework-agnostic IRGraph
            from app.ir.ir_graph import IRGraph
            ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
            
            # Apply graph optimizations
            from app.services.graph_engine import GraphOptimizer
            GraphOptimizer.simplify_graph(ir_graph)

            # Generate TensorFlow code
            from app.codegen.generators.registry import GeneratorRegistry
            compiler = GeneratorRegistry.get_generator("TensorFlow")
            generated_code = compiler.generate(ir_graph)
            
            # Save to Cache
            CachingService.set(cache_key, generated_code, expire_seconds=3600)
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="GENERATE_TENSORFLOW_CODE",
                resource_type="PROJECT",
                resource_id=str(project.id),
                ip_address=info.context.ip_address
            )
            return generated_code
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def generate_jax_code(
        self,
        project_id: strawberry.ID,
        info
    ) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.project import Project
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)
            
        if not project:
            raise Exception("Project not found.")

        # Check Cache
        from app.services.caching_service import CachingService
        cache_key = f"cache:project:jax:{project_id}"
        cached_code = CachingService.get(cache_key)
        if cached_code:
            return cached_code

        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        try:
            from app.services.validation_service import ValidationService
            sorted_nodes = ValidationService.validate_graph(nodes, edges)

            from app.services.shape_inference_service import ShapeInferenceService
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
            db.commit()

            from app.ir.ir_graph import IRGraph
            ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
            
            from app.services.graph_engine import GraphOptimizer
            GraphOptimizer.simplify_graph(ir_graph)

            from app.codegen.generators.registry import GeneratorRegistry
            compiler = GeneratorRegistry.get_generator("JAX")
            generated_code = compiler.generate(ir_graph)
            
            # Save to Cache
            CachingService.set(cache_key, generated_code, expire_seconds=3600)
            
            AuditService.log_action(
                db,
                user_id=user.id,
                action="GENERATE_JAX_CODE",
                resource_type="PROJECT",
                resource_id=str(project.id),
                ip_address=info.context.ip_address
            )
            return generated_code
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def generate_onnx_code(
        self,
        project_id: strawberry.ID,
        info
    ) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.project import Project
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)
            
        if not project:
            raise Exception("Project not found.")

        # Check Cache
        from app.services.caching_service import CachingService
        cache_key = f"cache:project:onnx:{project_id}"
        cached_code = CachingService.get(cache_key)
        if cached_code:
            return cached_code

        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        try:
            from app.services.validation_service import ValidationService
            sorted_nodes = ValidationService.validate_graph(nodes, edges)

            from app.services.shape_inference_service import ShapeInferenceService
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
            db.commit()

            from app.ir.ir_graph import IRGraph
            ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
            
            from app.services.graph_engine import GraphOptimizer
            GraphOptimizer.simplify_graph(ir_graph)

            from app.codegen.generators.registry import GeneratorRegistry
            compiler = GeneratorRegistry.get_generator("ONNX")
            generated_code = compiler.generate(ir_graph)
            
            # Save to Cache
            CachingService.set(cache_key, generated_code, expire_seconds=3600)
            
            AuditService.log_action(
                db,
                user_id=user.id,
                action="GENERATE_ONNX_CODE",
                resource_type="PROJECT",
                resource_id=str(project.id),
                ip_address=info.context.ip_address
            )
            return generated_code
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def export_onnx(
        self,
        project_id: strawberry.ID,
        info
    ) -> ExportArtifactType:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.project import Project
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)
            
        if not project:
            raise Exception("Project not found.")

        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        try:
            from app.services.validation_service import ValidationService
            sorted_nodes = ValidationService.validate_graph(nodes, edges)

            from app.services.shape_inference_service import ShapeInferenceService
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
            db.commit()

            from app.ir.ir_graph import IRGraph
            ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
            
            from app.services.graph_engine import GraphOptimizer
            GraphOptimizer.simplify_graph(ir_graph)

            from app.codegen.generators.registry import GeneratorRegistry
            compiler = GeneratorRegistry.get_generator("ONNX")
            generated_code = compiler.generate(ir_graph)
            
            # Execute generated Python code to construct ONNX model
            namespace = {}
            exec(generated_code, namespace)
            onnx_model = namespace["create_mlbuilder_onnx_model"]()

            # Ensure export directory exists
            import os
            export_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "exports", str(project.id)))
            os.makedirs(export_dir, exist_ok=True)
            artifact_path = os.path.join(export_dir, "model.onnx")

            # pyrefly: ignore [missing-import]
            import onnx
            onnx.save(onnx_model, artifact_path)

            # Compute SHA256 checksum of generated model.onnx
            import hashlib
            sha256 = hashlib.sha256()
            with open(artifact_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            checksum = sha256.hexdigest()

            # Record in Database
            from app.models.export_artifact import ExportArtifact
            from datetime import datetime
            artifact = db.query(ExportArtifact).filter(
                ExportArtifact.project_id == project.id,
                ExportArtifact.framework == "ONNX"
            ).first()

            if not artifact:
                artifact = ExportArtifact(
                    id=uuid.uuid4(),
                    project_id=project.id,
                    framework="ONNX",
                    artifact_path=artifact_path,
                    checksum=checksum
                )
                db.add(artifact)
            else:
                artifact.artifact_path = artifact_path
                artifact.checksum = checksum
                artifact.created_at = datetime.utcnow()
            db.commit()

            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="EXPORT_ONNX_MODEL",
                resource_type="PROJECT",
                resource_id=str(project.id),
                ip_address=info.context.ip_address
            )

            return ExportArtifactType(
                id=artifact.id,
                project_id=artifact.project_id,
                framework=artifact.framework,
                artifact_path=artifact.artifact_path,
                checksum=artifact.checksum,
                created_at=artifact.created_at
            )
        except Exception as e:
            db.rollback()
            raise Exception(str(e))


    @strawberry.mutation
    def validate_project_compilation(
        self,
        project_id: strawberry.ID,
        info
    ) -> CompilationValidationPayload:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        # Fetch project with role-sensitive checks
        from app.models.project import Project
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)
            
        if not project:
            raise Exception("Project not found.")

        # Fetch nodes and edges
        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        semantic_errors = []
        compatibility_errors = []
        compilation_errors = []
        generated_code = ""
        execution_logs = ""
        sorted_nodes = []

        # 1. Topological validation and Shape Inference
        try:
            from app.services.validation_service import ValidationService
            sorted_nodes = ValidationService.validate_graph(nodes, edges)
            
            # Infer shapes and save to node records
            from app.services.shape_inference_service import ShapeInferenceService
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
            db.commit()

            # 2. Semantic Checks
            semantic_errors = ValidationService.validate_semantics(sorted_nodes, edges)
        except Exception as dag_err:
            semantic_errors = [str(dag_err)]

        # 3. If graph structure passes, compile code and run framework/sandbox validations
        if sorted_nodes and not semantic_errors:
            try:
                # 3a. Framework Compatibility Validation
                compatibility_errors = ValidationService.validate_framework_compatibility(
                    sorted_nodes, edges, project.framework
                )

                # 3b. Code compilation using framework-agnostic IRGraph
                from app.ir.ir_graph import IRGraph
                ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
                
                from app.services.graph_engine import GraphOptimizer
                GraphOptimizer.simplify_graph(ir_graph)

                from app.codegen.generators.registry import GeneratorRegistry
                try:
                    compiler = GeneratorRegistry.get_generator(project.framework)
                    generated_code = compiler.generate(ir_graph)
                except ValueError:
                    generated_code = f"# Framework compiler for '{project.framework}' not supported."

                # 3c. Sandbox Compilation Execution Verification
                from app.services.compilation_sandbox import CompilationSandbox
                sandbox_res = CompilationSandbox.validate_compilation(
                    code=generated_code,
                    framework=project.framework,
                    project_id=str(project.id)
                )
                if not sandbox_res["success"]:
                    compilation_errors = sandbox_res["compilation_errors"]
                execution_logs = sandbox_res["logs"]

            except Exception as internal_err:
                compilation_errors = [f"Internal Compiler Error: {str(internal_err)}"]
                execution_logs = "Compilation crash inside backend compiler."
        else:
            execution_logs = "Compilation and sandbox check skipped due to topological or semantic shape verification errors."

        success = len(semantic_errors) == 0 and len(compilation_errors) == 0

        return CompilationValidationPayload(
            success=success,
            semantic_errors=semantic_errors,
            compatibility_errors=compatibility_errors,
            compilation_errors=compilation_errors,
            generated_code=generated_code,
            execution_logs=execution_logs
        )

    @strawberry.mutation
    def trigger_async_compilation(
        self,
        project_id: strawberry.ID,
        info
    ) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.project import Project
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)
            
        if not project:
            raise Exception("Project not found.")

        from app.tasks.tasks import async_compile_and_validate
        task = async_compile_and_validate.delay(str(proj_uuid), str(project.user_id))
        return task.id

    @strawberry.mutation
    def cancel_task(
        self,
        task_id: str,
        info
    ) -> bool:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        from app.tasks.celery_app import celery_app
        celery_app.control.revoke(task_id, terminate=True)
        return True

    @strawberry.mutation
    def create_dataset(
        self,
        name: str,
        dataset_type: str,
        filename: str,
        info,
        description: str | None = None
    ) -> DatasetUploadPayload:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        from app.services.dataset_service import DatasetService
        try:
            dataset, upload_url = DatasetService.create_dataset(
                db,
                user_id=user.id,
                name=name,
                dataset_type=dataset_type,
                filename=filename,
                description=description
            )
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="CREATE_DATASET",
                resource_type="DATASET",
                resource_id=str(dataset.id),
                details={"name": name, "dataset_type": dataset_type},
                ip_address=info.context.ip_address
            )
            
            dataset_gql = DatasetType(
                id=dataset.id,
                user_id=dataset.user_id,
                project_id=dataset.project_id,
                name=dataset.name,
                description=dataset.description,
                dataset_type=dataset.dataset_type,
                status=dataset.status,
                file_path=dataset.file_path,
                num_records=dataset.num_records,
                schema_metadata=dataset.schema_metadata,
                storage_path=dataset.storage_path,
                row_count=dataset.row_count,
                column_count=dataset.column_count,
                metadata_json=dataset.metadata_json,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at
            )
            return DatasetUploadPayload(dataset=dataset_gql, upload_url=upload_url)
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def upload_dataset(
        self,
        name: str,
        dataset_type: str,
        filename: str,
        info,
        description: str | None = None,
        project_id: strawberry.ID | None = None
    ) -> DatasetUploadPayload:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id) if project_id else None
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.services.dataset_service import DatasetService
        try:
            dataset, upload_url = DatasetService.create_dataset(
                db,
                user_id=user.id,
                name=name,
                dataset_type=dataset_type,
                filename=filename,
                description=description,
                project_id=proj_uuid
            )
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="CREATE_DATASET",
                resource_type="DATASET",
                resource_id=str(dataset.id),
                details={"name": name, "dataset_type": dataset_type},
                ip_address=info.context.ip_address
            )
            
            dataset_gql = DatasetType(
                id=dataset.id,
                user_id=dataset.user_id,
                project_id=dataset.project_id,
                name=dataset.name,
                description=dataset.description,
                dataset_type=dataset.dataset_type,
                status=dataset.status,
                file_path=dataset.file_path,
                num_records=dataset.num_records,
                schema_metadata=dataset.schema_metadata,
                storage_path=dataset.storage_path,
                row_count=dataset.row_count,
                column_count=dataset.column_count,
                metadata_json=dataset.metadata_json,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at
            )
            return DatasetUploadPayload(dataset=dataset_gql, upload_url=upload_url)
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def delete_dataset(
        self,
        id: strawberry.ID,
        info
    ) -> bool:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            dataset_uuid = uuid.UUID(id)
        except ValueError:
            raise Exception("Invalid dataset ID format.")

        from app.services.dataset_service import DatasetService
        try:
            res = DatasetService.delete_dataset(db, dataset_uuid, user.id)
            if res:
                # Log Audit trail
                AuditService.log_action(
                    db,
                    user_id=user.id,
                    action="DELETE_DATASET",
                    resource_type="DATASET",
                    resource_id=id,
                    ip_address=info.context.ip_address
                )
            return res
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def trigger_dataset_processing(
        self,
        dataset_id: strawberry.ID,
        info
    ) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor"], dataset_id=dataset_id)
        db = info.context.db
        try:
            dataset_uuid = uuid.UUID(dataset_id)
        except ValueError:
            raise Exception("Invalid dataset ID format.")

        from app.models.dataset import Dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
        if not dataset:
            raise Exception("Dataset not found.")

        from app.services.dataset_service import DatasetService
        try:
            task_id = DatasetService.trigger_dataset_processing(db, dataset_uuid, dataset.user_id)
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="TRIGGER_DATASET_PROCESSING",
                resource_type="DATASET",
                resource_id=str(dataset_uuid),
                ip_address=info.context.ip_address
            )
            return task_id
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def trigger_training_job(
        self,
        project_id: strawberry.ID,
        epochs: int,
        info,
        dataset_id: strawberry.ID | None = None
    ) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id, dataset_id=dataset_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
            ds_uuid = uuid.UUID(dataset_id) if dataset_id else None
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.models.project import Project
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)
            
        if not project:
            raise Exception("Project not found.")

        try:
            from app.services.cloud_training_service import CloudTrainingService
            _, task_or_job_id = CloudTrainingService.trigger_training_job(
                db,
                project_id=proj_uuid,
                epochs=epochs,
                dataset_id=ds_uuid
            )
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="TRIGGER_TRAINING_JOB",
                resource_type="PROJECT",
                resource_id=str(proj_uuid),
                details={"epochs": epochs, "dataset_id": str(ds_uuid) if ds_uuid else None},
                ip_address=info.context.ip_address
            )
            return task_or_job_id
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def cancel_training_job(
        self,
        training_job_id: strawberry.ID,
        info
    ) -> bool:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            job_uuid = uuid.UUID(training_job_id)
        except ValueError:
            raise Exception("Invalid training job ID format.")

        from app.models.training_job import TrainingJob
        job = db.query(TrainingJob).filter(TrainingJob.id == job_uuid).first()
        if not job:
            raise Exception("Training job not found.")

        # Check project ownership/permissions
        verify_role_and_ownership(info, ["admin", "editor"], project_id=job.project_id)

        if job.status in {"COMPLETED", "FAILED", "CANCELLED"}:
            raise Exception(f"Cannot cancel a training job in {job.status} state.")

        # Revoke the Celery task (hard cancellation)
        if job.celery_task_id:
            from app.tasks.celery_app import celery_app
            celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGKILL")

        # Soft cancellation: update status to CANCELLED in DB
        job.status = "CANCELLED"
        db.commit()

        # Publish WebSocket event
        from app.services.event_dispatcher import EventDispatcher
        EventDispatcher.get_redis().publish(
            "mlbuilder:project:training",
            f'{{"type": "TrainingCancelled", "training_job_id": "{training_job_id}", "status": "CANCELLED"}}'
        )

        # Log audit trail
        AuditService.log_action(
            db,
            user_id=user.id,
            action="CANCEL_TRAINING_JOB",
            resource_type="PROJECT",
            resource_id=str(job.project_id),
            details={"training_job_id": str(job_uuid)},
            ip_address=info.context.ip_address
        )

        return True

    @strawberry.mutation
    def benchmark_project(
        self,
        project_id: strawberry.ID,
        info
    ) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        # Fetch and verify role-sensitive project checks
        from app.models.project import Project
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)
            
        if not project:
            raise Exception("Project not found.")

        # Check Cache
        from app.services.caching_service import CachingService
        cache_key = f"cache:project:benchmark:{project_id}"
        cached_res = CachingService.get(cache_key)
        if cached_res:
            return cached_res

        # Compile code dynamically to feed to benchmarker
        from app.models.node import Node
        from app.models.edge import Edge
        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        try:
            # Run topological sorting and shape inference
            from app.services.validation_service import ValidationService
            sorted_nodes = ValidationService.validate_graph(nodes, edges)

            from app.services.shape_inference_service import ShapeInferenceService
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
            db.commit()

            # Compile PyTorch code using IRGraph
            from app.ir.ir_graph import IRGraph
            ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
            
            from app.services.graph_engine import GraphOptimizer
            GraphOptimizer.simplify_graph(ir_graph)

            from app.codegen.generators.registry import GeneratorRegistry
            compiler = GeneratorRegistry.get_generator("PyTorch")
            generated_code = compiler.generate(ir_graph)

            # Invoke sandboxed dynamic benchmarking
            from app.services.benchmarking_service import BenchmarkingService
            res = BenchmarkingService.benchmark_compiled_model(
                proj_uuid, 
                generated_code, 
                project.framework, 
                sorted_nodes
            )
            
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="BENCHMARK_PROJECT",
                resource_type="PROJECT",
                resource_id=str(proj_uuid),
                ip_address=info.context.ip_address
            )
            
            import json
            # Save to Cache
            try:
                CachingService.set(cache_key, json.dumps(res), expire_seconds=3600)
            except Exception:
                pass
                
            return json.dumps(res)
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def apply_architecture_template(self, info, project_id: strawberry.ID, prompt: str) -> ProjectType:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.project import Project
        from app.models.node import Node
        from app.models.edge import Edge
        project = db.query(Project).filter(Project.id == proj_uuid).first()
        if not project:
            raise Exception("Project not found.")

        # 1. Clear existing nodes and edges
        db.query(Node).filter(Node.project_id == proj_uuid).delete()
        db.query(Edge).filter(Edge.project_id == proj_uuid).delete()

        # 2. Generate template layout
        from app.services.template_generator import ArchitectureTemplateGenerator
        template = ArchitectureTemplateGenerator.generate(prompt)

        # 3. Create new Node models
        for n in template["nodes"]:
            db_node = Node(
                id=uuid.UUID(n["id"]),
                project_id=proj_uuid,
                type=n["type"],
                label=n["label"],
                config=n["config"],
                position_x=n["position_x"],
                position_y=n["position_y"],
                input_shape=n["input_shape"],
                output_shape=n["output_shape"]
            )
            db.add(db_node)

        # 4. Create new Edge models
        for e in template["edges"]:
            db_edge = Edge(
                id=uuid.UUID(e["id"]),
                project_id=proj_uuid,
                from_node_id=uuid.UUID(e["from_node_id"]),
                to_node_id=uuid.UUID(e["to_node_id"]),
                input_shape=e["input_shape"],
                output_shape=e["output_shape"]
            )
            db.add(db_edge)

        # Commit changes
        db.commit()
        db.refresh(project)

        # Log Audit event
        AuditService.log_action(
            db,
            user_id=user.id,
            action="APPLY_TEMPLATE",
            resource_type="PROJECT",
            resource_id=str(proj_uuid),
            details={"prompt": prompt, "template_name": template["name"]},
            ip_address=info.context.ip_address
        )

        return ProjectType(
            id=project.id,
            user_id=project.user_id,
            name=project.name,
            description=project.description,
            framework=project.framework,
            is_public=project.is_public,
            thumbnail_url=project.thumbnail_url,
            created_at=project.created_at,
            updated_at=project.updated_at
        )

    @strawberry.mutation
    def update_user_role(self, info, user_id: strawberry.ID, role: str) -> UserType:
        admin_user = verify_role_and_ownership(info, ["admin"])
        if role not in {"admin", "editor", "viewer"}:
            raise Exception("Invalid role. Supported roles: admin, editor, viewer.")
            
        db = info.context.db
        try:
            target_uuid = uuid.UUID(user_id)
        except ValueError:
            raise Exception("Invalid user ID format.")
            
        from app.models.user import User
        target_user = db.query(User).filter(User.id == target_uuid).first()
        if not target_user:
            raise Exception("User not found.")
            
        old_role = target_user.role
        target_user.role = role
        db.commit()
        
        # Log Audit event
        AuditService.log_action(
            db,
            user_id=admin_user.id,
            action="UPDATE_USER_ROLE",
            resource_type="USER",
            resource_id=str(target_user.id),
            details={"old_role": old_role, "new_role": role},
            ip_address=info.context.ip_address
        )
        
        return UserType(
            id=target_user.id,
            email=target_user.email,
            username=target_user.username,
            preferences=target_user.preferences or {},
            role=target_user.role,
            created_at=target_user.created_at,
            updated_at=target_user.updated_at
        )

    @strawberry.mutation
    def deploy_model(self, info, artifact_id: strawberry.ID, target: str) -> DeploymentType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            art_uuid = uuid.UUID(artifact_id)
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.models.model_artifact import ModelArtifact
        artifact = db.query(ModelArtifact).filter(ModelArtifact.id == art_uuid).first()
        if not artifact:
            raise Exception("Model artifact not found.")

        verify_role_and_ownership(info, ["admin", "editor"], project_id=str(artifact.project_id))

        from app.services.deployment_service import DeploymentService
        d = DeploymentService.deploy_artifact(db, art_uuid, target)

        AuditService.log_action(
            db,
            user_id=user.id,
            action="DEPLOY_MODEL",
            resource_type="DEPLOYMENT",
            resource_id=str(d.id),
            details={"artifact_id": artifact_id, "target": target},
            ip_address=info.context.ip_address
        )

        return DeploymentType(
            id=d.id,
            project_id=d.project_id,
            model_artifact_id=d.model_artifact_id,
            target=d.target,
            status=d.status,
            endpoint_url=d.endpoint_url,
            created_at=d.created_at,
            updated_at=d.updated_at
        )

    @strawberry.mutation
    def predict_deployment(self, info, deployment_id: strawberry.ID, input_data: strawberry.scalars.JSON) -> strawberry.scalars.JSON:
        verify_role_and_ownership(info, ["admin", "editor", "viewer"])
        db = info.context.db
        try:
            dep_uuid = uuid.UUID(deployment_id)
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.models.deployment import Deployment
        d = db.query(Deployment).filter(Deployment.id == dep_uuid).first()
        if not d:
            raise Exception("Deployment not found.")

        verify_role_and_ownership(info, ["admin", "editor", "viewer"], project_id=str(d.project_id))

        if d.status != "ACTIVE":
            raise Exception(f"Deployment is not ACTIVE. Current status: {d.status}")

        import time
        start_time = time.perf_counter()
        has_error = False
        try:
            from app.services.inference_service import InferenceService
            result = InferenceService.execute_prediction(db, d.model_artifact_id, input_data)
        except Exception as e:
            has_error = True
            raise Exception(str(e))
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            from app.services.deployment_service import DeploymentService
            DeploymentService.record_metrics(db, d.id, latency_ms, has_error)

        return result

    @strawberry.mutation
    def export_deployment_package(self, info, artifact_id: strawberry.ID) -> str:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            art_uuid = uuid.UUID(artifact_id)
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.models.model_artifact import ModelArtifact
        artifact = db.query(ModelArtifact).filter(ModelArtifact.id == art_uuid).first()
        if not artifact:
            raise Exception("Model artifact not found.")

        verify_role_and_ownership(info, ["admin", "editor"], project_id=str(artifact.project_id))

        from app.services.deployment_service import DeploymentService
        zip_url = DeploymentService.export_prediction_package(db, art_uuid)

        AuditService.log_action(
            db,
            user_id=user.id,
            action="EXPORT_DEPLOYMENT_PACKAGE",
            resource_type="MODEL_ARTIFACT",
            resource_id=artifact_id,
            ip_address=info.context.ip_address
        )

        return zip_url

    @strawberry.mutation
    def create_experiment(self, info, project_id: strawberry.ID, name: str, description: str | None = None) -> ExperimentType:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.models.experiment import Experiment
        experiment = Experiment(
            id=uuid.uuid4(),
            project_id=proj_uuid,
            name=name,
            description=description,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(experiment)
        db.commit()

        AuditService.log_action(
            db,
            user_id=user.id,
            action="CREATE_EXPERIMENT",
            resource_type="EXPERIMENT",
            resource_id=str(experiment.id),
            details={"name": name},
            ip_address=info.context.ip_address
        )

        return ExperimentType(
            id=experiment.id,
            project_id=experiment.project_id,
            name=experiment.name,
            description=experiment.description,
            created_at=experiment.created_at,
            updated_at=experiment.updated_at
        )

    @strawberry.mutation
    def create_dataset_version(
        self, 
        info, 
        dataset_id: strawberry.ID, 
        version_number: str, 
        storage_path: str,
        row_count: int = 0,
        column_count: int = 0,
        metadata_json: strawberry.scalars.JSON | None = None
    ) -> DatasetVersionType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            ds_uuid = uuid.UUID(dataset_id)
        except ValueError:
            raise Exception("Invalid dataset ID format.")

        from app.models.dataset import Dataset
        dataset = db.query(Dataset).filter(Dataset.id == ds_uuid).first()
        if not dataset:
            raise Exception("Dataset not found.")

        if user.role != "admin" and dataset.user_id != user.id:
            raise Exception("Forbidden: You do not have permission to version this dataset.")

        from app.models.dataset_version import DatasetVersion
        version = DatasetVersion(
            id=uuid.uuid4(),
            dataset_id=ds_uuid,
            version_number=version_number,
            storage_path=storage_path,
            row_count=row_count,
            column_count=column_count,
            metadata_json=metadata_json,
            created_at=datetime.utcnow()
        )
        db.add(version)
        
        # update dataset metadata
        dataset.storage_path = storage_path
        dataset.row_count = row_count
        dataset.column_count = column_count
        if metadata_json:
            dataset.metadata_json = metadata_json
        dataset.status = "READY"
        
        db.commit()

        AuditService.log_action(
            db,
            user_id=user.id,
            action="CREATE_DATASET_VERSION",
            resource_type="DATASET_VERSION",
            resource_id=str(version.id),
            details={"version_number": version_number},
            ip_address=info.context.ip_address
        )

        return DatasetVersionType(
            id=version.id,
            dataset_id=version.dataset_id,
            version_number=version.version_number,
            storage_path=version.storage_path,
            row_count=version.row_count,
            column_count=version.column_count,
            metadata_json=version.metadata_json,
            created_at=version.created_at
        )

    @strawberry.mutation
    def add_run_to_experiment(self, info, experiment_id: strawberry.ID, run_id: strawberry.ID) -> TrainingRunType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            exp_uuid = uuid.UUID(experiment_id)
            run_uuid = uuid.UUID(run_id)
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.models.experiment import Experiment
        from app.models.training_run import TrainingRun
        
        experiment = db.query(Experiment).filter(Experiment.id == exp_uuid).first()
        if not experiment:
            raise Exception("Experiment not found.")

        verify_role_and_ownership(info, ["admin", "editor"], project_id=str(experiment.project_id))

        run = db.query(TrainingRun).filter(TrainingRun.id == run_uuid).first()
        if not run:
            raise Exception("Training run not found.")

        if run.project_id != experiment.project_id:
            raise Exception("Cannot group a run from a different project into this experiment.")

        run.experiment_id = experiment.id
        db.commit()

        AuditService.log_action(
            db,
            user_id=user.id,
            action="ADD_RUN_TO_EXPERIMENT",
            resource_type="TRAINING_RUN",
            resource_id=str(run.id),
            details={"experiment_id": experiment_id},
            ip_address=info.context.ip_address
        )

        return TrainingRunType(
            id=run.id,
            project_id=run.project_id,
            training_job_id=run.training_job_id,
            accuracy=run.accuracy,
            loss=run.loss,
            metrics_json=run.metrics_json,
            config_json=run.config_json,
            created_at=run.created_at,
            updated_at=run.updated_at
        )

    @strawberry.mutation
    def rollback_deployment(self, info, deployment_id: strawberry.ID, target_version: str) -> DeploymentType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            dep_uuid = uuid.UUID(deployment_id)
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.models.deployment import Deployment
        d = db.query(Deployment).filter(Deployment.id == dep_uuid).first()
        if not d:
            raise Exception("Deployment not found.")

        verify_role_and_ownership(info, ["admin", "editor"], project_id=str(d.project_id))

        from app.services.deployment_service import DeploymentService
        d_rolled = DeploymentService.rollback_deployment(db, dep_uuid, target_version)

        AuditService.log_action(
            db,
            user_id=user.id,
            action="ROLLBACK_DEPLOYMENT",
            resource_type="DEPLOYMENT",
            resource_id=str(d_rolled.id),
            details={"target_version": target_version},
            ip_address=info.context.ip_address
        )

        return DeploymentType(
            id=d_rolled.id,
            project_id=d_rolled.project_id,
            model_artifact_id=d_rolled.model_artifact_id,
            target=d_rolled.target,
            status=d_rolled.status,
            endpoint_url=d_rolled.endpoint_url,
            created_at=d_rolled.created_at,
            updated_at=d_rolled.updated_at
        )

    @strawberry.mutation
    def generate_architecture(self, info, project_id: strawberry.ID, prompt: str) -> ProjectType:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")
            
        try:
            from app.services.copilot.copilot_service import CopilotService
            CopilotService.generate_architecture(db, proj_uuid, prompt)
            
            # Re-fetch project
            from app.models.project import Project
            project = db.query(Project).filter(Project.id == proj_uuid).first()
            if not project:
                raise Exception("Project not found after generation.")
                
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="GENERATE_ARCHITECTURE",
                resource_type="PROJECT",
                resource_id=project_id,
                details={"prompt": prompt},
                ip_address=info.context.ip_address
            )
            
            return ProjectType(
                id=project.id,
                user_id=project.user_id,
                name=project.name,
                description=project.description,
                framework=project.framework,
                is_public=project.is_public,
                thumbnail_url=project.thumbnail_url,
                created_at=project.created_at,
                updated_at=project.updated_at
            )
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def modify_architecture(self, info, project_id: strawberry.ID, prompt: str) -> ProjectType:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")
            
        try:
            from app.services.copilot.copilot_service import CopilotService
            CopilotService.modify_architecture(db, proj_uuid, prompt)
            
            # Re-fetch project
            from app.models.project import Project
            project = db.query(Project).filter(Project.id == proj_uuid).first()
            if not project:
                raise Exception("Project not found after modification.")
                
            # Log Audit trail
            AuditService.log_action(
                db,
                user_id=user.id,
                action="MODIFY_ARCHITECTURE",
                resource_type="PROJECT",
                resource_id=project_id,
                details={"prompt": prompt},
                ip_address=info.context.ip_address
            )
            
            return ProjectType(
                id=project.id,
                user_id=project.user_id,
                name=project.name,
                description=project.description,
                framework=project.framework,
                is_public=project.is_public,
                thumbnail_url=project.thumbnail_url,
                created_at=project.created_at,
                updated_at=project.updated_at
            )
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def register_model(
        self,
        info,
        project_id: strawberry.ID,
        name: str,
        description: str | None = None
    ) -> RegisteredModelType:
        user = verify_role_and_ownership(info, ["admin", "editor"], project_id=project_id)
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        from app.services.model_registry_service import ModelRegistryService
        try:
            model = ModelRegistryService.register_model(db, proj_uuid, name, description)
            
            AuditService.log_action(
                db,
                user_id=user.id,
                action="REGISTER_MODEL",
                resource_type="REGISTERED_MODEL",
                resource_id=str(model.id),
                details={"name": name, "project_id": project_id},
                ip_address=info.context.ip_address
            )
            return RegisteredModelType(
                id=model.id,
                project_id=model.project_id,
                name=model.name,
                description=model.description,
                created_at=model.created_at,
                updated_at=model.updated_at
            )
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def create_model_version(
        self,
        info,
        model_id: strawberry.ID,
        version: str,
        description: str | None = None,
        artifact_id: strawberry.ID | None = None,
        metrics: strawberry.scalars.JSON | None = None,
        config: strawberry.scalars.JSON | None = None,
        compiler_output: str | None = None
    ) -> ModelVersionType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            model_uuid = uuid.UUID(model_id)
            art_uuid = uuid.UUID(artifact_id) if artifact_id else None
        except ValueError:
            raise Exception("Invalid UUID format.")

        from app.models.registered_model import RegisteredModel
        model = db.query(RegisteredModel).filter(RegisteredModel.id == model_uuid).first()
        if not model:
            raise Exception("Registered model not found.")

        verify_role_and_ownership(info, ["admin", "editor"], project_id=str(model.project_id))

        from app.services.model_registry_service import ModelRegistryService
        try:
            mv = ModelRegistryService.create_version(
                db=db,
                model_id=model_uuid,
                version=version,
                description=description,
                artifact_id=art_uuid,
                metrics=metrics,
                config=config,
                compiler_output=compiler_output
            )

            AuditService.log_action(
                db,
                user_id=user.id,
                action="CREATE_MODEL_VERSION",
                resource_type="MODEL_VERSION",
                resource_id=str(mv.id),
                details={"version": version, "model_id": model_id},
                ip_address=info.context.ip_address
            )
            return ModelVersionType(
                id=mv.id,
                model_id=mv.model_id,
                version=mv.version,
                description=mv.description,
                status=mv.status,
                model_artifact_id=mv.model_artifact_id,
                metrics=mv.metrics,
                config=mv.config,
                compiler_output=mv.compiler_output,
                created_at=mv.created_at,
                updated_at=mv.updated_at
            )
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def promote_model_version(
        self,
        info,
        version_id: strawberry.ID,
        status: str
    ) -> ModelVersionType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            ver_uuid = uuid.UUID(version_id)
        except ValueError:
            raise Exception("Invalid version ID format.")

        from app.models.model_version import ModelVersion
        from app.models.registered_model import RegisteredModel
        mv = db.query(ModelVersion).filter(ModelVersion.id == ver_uuid).first()
        if not mv:
            raise Exception("Model version not found.")

        model = db.query(RegisteredModel).filter(RegisteredModel.id == mv.model_id).first()
        if not model:
            raise Exception("Registered model not found.")

        verify_role_and_ownership(info, ["admin", "editor"], project_id=str(model.project_id))

        from app.services.model_registry_service import ModelRegistryService
        try:
            mv_promoted = ModelRegistryService.promote_version(db, ver_uuid, status)

            AuditService.log_action(
                db,
                user_id=user.id,
                action="PROMOTE_MODEL_VERSION",
                resource_type="MODEL_VERSION",
                resource_id=str(mv_promoted.id),
                details={"status": status},
                ip_address=info.context.ip_address
            )
            return ModelVersionType(
                id=mv_promoted.id,
                model_id=mv_promoted.model_id,
                version=mv_promoted.version,
                description=mv_promoted.description,
                status=mv_promoted.status,
                model_artifact_id=mv_promoted.model_artifact_id,
                metrics=mv_promoted.metrics,
                config=mv_promoted.config,
                compiler_output=mv_promoted.compiler_output,
                created_at=mv_promoted.created_at,
                updated_at=mv_promoted.updated_at
            )
        except Exception as e:
            raise Exception(str(e))

    @strawberry.mutation
    def stop_deployment(self, info, deployment_id: strawberry.ID) -> DeploymentType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            dep_uuid = uuid.UUID(deployment_id)
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.models.deployment import Deployment
        dep = db.query(Deployment).filter(Deployment.id == dep_uuid).first()
        if not dep:
            raise Exception("Deployment not found.")

        verify_role_and_ownership(info, ["admin", "editor"], project_id=str(dep.project_id))

        from app.services.deployment_service import DeploymentService
        d = DeploymentService.stop_deployment(db, dep_uuid)

        AuditService.log_action(
            db,
            user_id=user.id,
            action="STOP_DEPLOYMENT",
            resource_type="DEPLOYMENT",
            resource_id=str(d.id),
            ip_address=info.context.ip_address
        )

        return DeploymentType(
            id=d.id,
            project_id=d.project_id,
            model_artifact_id=d.model_artifact_id,
            target=d.target,
            status=d.status,
            endpoint_url=d.endpoint_url,
            created_at=d.created_at,
            updated_at=d.updated_at
        )

    @strawberry.mutation
    def create_workflow(
        self,
        info,
        project_id: strawberry.ID | None,
        name: str,
        trigger_event: str,
        action_type: str,
        config: strawberry.scalars.JSON
    ) -> WorkflowType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db

        proj_uuid = uuid.UUID(project_id) if project_id else None
        if proj_uuid:
            verify_role_and_ownership(info, ["admin", "editor"], project_id=str(proj_uuid))

        from app.services.workflow_service import WorkflowService
        w = WorkflowService.create_workflow(
            db=db,
            project_id=proj_uuid,
            name=name,
            trigger_event=trigger_event,
            action_type=action_type,
            config=config
        )

        AuditService.log_action(
            db,
            user_id=user.id,
            action="CREATE_WORKFLOW",
            resource_type="WORKFLOW",
            resource_id=str(w.id),
            details={"name": name, "trigger_event": trigger_event, "action_type": action_type},
            ip_address=info.context.ip_address
        )

        return WorkflowType(
            id=w.id,
            project_id=w.project_id,
            name=w.name,
            trigger_event=w.trigger_event,
            action_type=w.action_type,
            config=w.config,
            is_active=w.is_active,
            created_at=w.created_at,
            updated_at=w.updated_at
        )

    @strawberry.mutation
    def delete_workflow(self, info, workflow_id: strawberry.ID) -> bool:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        try:
            wf_uuid = uuid.UUID(workflow_id)
        except ValueError:
            raise Exception("Invalid ID format.")

        from app.services.workflow_service import WorkflowService
        wf = WorkflowService.get_workflow(db, wf_uuid)
        if not wf:
            raise Exception("Workflow not found.")

        if wf.project_id:
            verify_role_and_ownership(info, ["admin", "editor"], project_id=str(wf.project_id))

        success = WorkflowService.delete_workflow(db, wf_uuid)
        if success:
            AuditService.log_action(
                db,
                user_id=user.id,
                action="DELETE_WORKFLOW",
                resource_type="WORKFLOW",
                resource_id=str(wf_uuid),
                ip_address=info.context.ip_address
            )

        return success

    @strawberry.mutation
    def execute_notebook_cell(
        self,
        info,
        project_id: strawberry.ID,
        code: str
    ) -> NotebookCellResultType:
        user = verify_role_and_ownership(info, ["admin", "editor"])
        db = info.context.db
        
        res = NotebookExecutionService.execute_cell(
            db=db,
            user=user,
            project_id=project_id,
            code=code,
            timeout=5
        )
        return NotebookCellResultType(
            success=res["success"],
            stdout=res["stdout"],
            stderr=res["stderr"],
            execution_time_ms=res["execution_time_ms"]
        )

# Import custom GraphQL security extensions
from app.graphql.extensions.depth_limiter import GraphQLDepthLimiter
from app.graphql.extensions.cost_limiter import GraphQLCostLimiter

schema = strawberry.Schema(
    query=Query, 
    mutation=Mutation,
    extensions=[
        GraphQLDepthLimiter(max_depth=4),
        GraphQLCostLimiter(max_cost=50)
    ]
)
