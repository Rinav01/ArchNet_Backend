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
from app.auth.rbac import verify_role_and_ownership
from app.services.audit_service import AuditService
from app.models.audit_log import AuditLog

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
            from app.codegen.pytorch_generator import PyTorchGenerator
            compiler = PyTorchGenerator()
            generated_code = compiler.compile(ir_graph)
            
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
            from app.codegen.pytorch.generator import PyTorchCompiler
            compiler = PyTorchCompiler()
            generated_code = compiler.compile(ir_graph)
            
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
            from app.codegen.tensorflow.compiler import TensorFlowCompiler
            compiler = TensorFlowCompiler()
            generated_code = compiler.compile(ir_graph)
            
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

            from app.codegen.jax.compiler import JAXCompiler
            compiler = JAXCompiler()
            generated_code = compiler.compile(ir_graph)
            
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

            from app.codegen.onnx.compiler import ONNXCompiler
            compiler = ONNXCompiler()
            generated_code = compiler.compile(ir_graph)
            
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

                framework_str = project.framework.lower().strip()
                if "pytorch" in framework_str or "torch" in framework_str:
                    from app.codegen.pytorch.generator import PyTorchCompiler
                    compiler = PyTorchCompiler()
                    generated_code = compiler.compile(ir_graph)
                elif "tensorflow" in framework_str or "keras" in framework_str:
                    from app.codegen.tensorflow.compiler import TensorFlowCompiler
                    compiler = TensorFlowCompiler()
                    generated_code = compiler.compile(ir_graph)
                elif "jax" in framework_str or "flax" in framework_str:
                    from app.codegen.jax.compiler import JAXCompiler
                    compiler = JAXCompiler()
                    generated_code = compiler.compile(ir_graph)
                elif "onnx" in framework_str:
                    from app.codegen.onnx.compiler import ONNXCompiler
                    compiler = ONNXCompiler()
                    generated_code = compiler.compile(ir_graph)
                else:
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

            from app.codegen.pytorch.generator import PyTorchCompiler
            compiler = PyTorchCompiler()
            generated_code = compiler.compile(ir_graph)

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
