import strawberry
import uuid
from typing import List

from app.graphql.types.user_type import UserType
from app.graphql.types.project_type import ProjectType
from app.graphql.types.node_type import NodeType, PositionInput
from app.graphql.types.edge_type import EdgeType
from app.graphql.types.auth_payload import AuthPayload

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
            created_at=user.created_at,
            updated_at=user.updated_at
        )

    @strawberry.field
    def project(self, info, id: strawberry.ID) -> ProjectType | None:
        user = info.context.current_user
        if not user:
            raise Exception("Not authenticated.")
        
        db = info.context.db
        try:
            project_uuid = uuid.UUID(id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        project = ProjectService.get_project(db, project_uuid, user_id=user.id)
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
        user = info.context.current_user
        if not user:
            raise Exception("Not authenticated.")
        
        db = info.context.db
        projects = ProjectService.list_projects(db, user_id=user.id, limit=limit, offset=offset)
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
            token = AuthService.login(db, email, password)[0]
            
            user_gql = UserType(
                id=user.id,
                email=user.email,
                username=user.username,
                preferences=user.preferences or {},
                created_at=user.created_at,
                updated_at=user.updated_at
            )
            return AuthPayload(token=token, user=user_gql)
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
            token, user = AuthService.login(db, email, password)
            user_gql = UserType(
                id=user.id,
                email=user.email,
                username=user.username,
                preferences=user.preferences or {},
                created_at=user.created_at,
                updated_at=user.updated_at
            )
            return AuthPayload(token=token, user=user_gql)
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
        user = info.context.current_user
        if not user:
            raise Exception("Not authenticated.")
            
        db = info.context.db
        project = ProjectService.create_project(
            db, 
            user_id=user.id, 
            name=name, 
            description=description, 
            framework=framework
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
        user = info.context.current_user
        if not user:
            raise Exception("Not authenticated.")
            
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        # Confirm ownership
        ProjectService.get_project(db, proj_uuid, user_id=user.id)
        
        node = ProjectService.add_node(
            db, 
            project_id=proj_uuid, 
            node_type=type, 
            label=label, 
            position_x=position.x, 
            position_y=position.y, 
            config=config
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
        user = info.context.current_user
        if not user:
            raise Exception("Not authenticated.")
            
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
            from_uuid = uuid.UUID(from_node_id)
            to_uuid = uuid.UUID(to_node_id)
        except ValueError:
            raise Exception("Invalid UUID format for IDs.")

        # Confirm ownership
        ProjectService.get_project(db, proj_uuid, user_id=user.id)
        
        try:
            edge = ProjectService.add_edge(db, proj_uuid, from_uuid, to_uuid)
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
        user = info.context.current_user
        if not user:
            raise Exception("Not authenticated.")
            
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
            node_uuid = uuid.UUID(node_id)
        except ValueError:
            raise Exception("Invalid UUID format.")

        # Confirm ownership
        ProjectService.get_project(db, proj_uuid, user_id=user.id)
        return ProjectService.delete_node(db, proj_uuid, node_uuid)

    @strawberry.mutation
    def delete_edge(
        self,
        project_id: strawberry.ID,
        edge_id: strawberry.ID,
        info
    ) -> bool:
        user = info.context.current_user
        if not user:
            raise Exception("Not authenticated.")
            
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
            edge_uuid = uuid.UUID(edge_id)
        except ValueError:
            raise Exception("Invalid UUID format.")

        # Confirm ownership
        ProjectService.get_project(db, proj_uuid, user_id=user.id)
        return ProjectService.delete_edge(db, proj_uuid, edge_uuid)

    @strawberry.mutation
    def generate_pytorch_code(
        self,
        project_id: strawberry.ID,
        info
    ) -> str:
        user = info.context.current_user
        if not user:
            raise Exception("Not authenticated.")
            
        db = info.context.db
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            raise Exception("Invalid project ID format.")

        # Fetch project with ownership verification
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

            # Persist computed shapes back to database
            db.commit()

            # Generate PyTorch code
            from app.codegen.pytorch.generator import PyTorchGenerator
            code = PyTorchGenerator.generate(project, sorted_nodes)
            return code
        except ValueError as e:
            raise Exception(str(e))

schema = strawberry.Schema(query=Query, mutation=Mutation)
