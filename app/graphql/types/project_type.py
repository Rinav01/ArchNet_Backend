import strawberry
import uuid
from datetime import datetime
from app.graphql.types.node_type import NodeType
from app.graphql.types.edge_type import EdgeType
from app.graphql.types.execution_plan_type import ExecutionPlanType

@strawberry.type
class ProjectType:
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    description: str | None
    framework: str
    is_public: bool
    thumbnail_url: str | None
    created_at: datetime
    updated_at: datetime

    @strawberry.field
    def nodes(self, info) -> list[NodeType]:
        pid = str(self.id)
        from app.services.caching_service import CachingService
        import json
        
        cached_nodes = CachingService.get(f"cache:project:nodes:{pid}")
        if cached_nodes:
            try:
                nodes_data = json.loads(cached_nodes)
                return [
                    NodeType(
                        id=uuid.UUID(n["id"]),
                        project_id=uuid.UUID(n["project_id"]),
                        type=n["type"],
                        label=n["label"],
                        position_x=n["position_x"],
                        position_y=n["position_y"],
                        config=n["config"],
                        input_shape=n["input_shape"],
                        output_shape=n["output_shape"],
                        created_at=datetime.fromisoformat(n["created_at"]) if n.get("created_at") else None,
                        updated_at=datetime.fromisoformat(n["updated_at"]) if n.get("updated_at") else None
                    ) for n in nodes_data
                ]
            except Exception:
                pass

        db = info.context.db
        from app.models.node import Node
        db_nodes = db.query(Node).filter(Node.project_id == self.id).all()
        
        try:
            nodes_to_cache = [
                {
                    "id": str(n.id),
                    "project_id": str(n.project_id),
                    "type": n.type,
                    "label": n.label,
                    "position_x": n.position_x,
                    "position_y": n.position_y,
                    "config": n.config,
                    "input_shape": n.input_shape,
                    "output_shape": n.output_shape,
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                    "updated_at": n.updated_at.isoformat() if n.updated_at else None
                } for n in db_nodes
            ]
            CachingService.set(f"cache:project:nodes:{pid}", json.dumps(nodes_to_cache), expire_seconds=3600)
        except Exception:
            pass

        return [
            NodeType(
                id=n.id,
                project_id=n.project_id,
                type=n.type,
                label=n.label,
                position_x=n.position_x,
                position_y=n.position_y,
                config=n.config,
                input_shape=n.input_shape,
                output_shape=n.output_shape,
                created_at=n.created_at,
                updated_at=n.updated_at
            ) for n in db_nodes
        ]

    @strawberry.field
    def edges(self, info) -> list[EdgeType]:
        pid = str(self.id)
        from app.services.caching_service import CachingService
        import json
        
        cached_edges = CachingService.get(f"cache:project:edges:{pid}")
        if cached_edges:
            try:
                edges_data = json.loads(cached_edges)
                return [
                    EdgeType(
                        id=uuid.UUID(e["id"]),
                        project_id=uuid.UUID(e["project_id"]),
                        from_node_id=uuid.UUID(e["from_node_id"]),
                        to_node_id=uuid.UUID(e["to_node_id"]),
                        input_shape=e["input_shape"],
                        output_shape=e["output_shape"],
                        created_at=datetime.fromisoformat(e["created_at"]) if e.get("created_at") else None
                    ) for e in edges_data
                ]
            except Exception:
                pass

        db = info.context.db
        from app.models.edge import Edge
        db_edges = db.query(Edge).filter(Edge.project_id == self.id).all()
        
        try:
            edges_to_cache = [
                {
                    "id": str(e.id),
                    "project_id": str(e.project_id),
                    "from_node_id": str(e.from_node_id),
                    "to_node_id": str(e.to_node_id),
                    "input_shape": e.input_shape,
                    "output_shape": e.output_shape,
                    "created_at": e.created_at.isoformat() if e.created_at else None
                } for e in db_edges
            ]
            CachingService.set(f"cache:project:edges:{pid}", json.dumps(edges_to_cache), expire_seconds=3600)
        except Exception:
            pass

        return [
            EdgeType(
                id=e.id,
                project_id=e.project_id,
                from_node_id=e.from_node_id,
                to_node_id=e.to_node_id,
                input_shape=e.input_shape,
                output_shape=e.output_shape,
                created_at=e.created_at
            ) for e in db_edges
        ]

    @strawberry.field
    def total_parameter_count(self, info) -> int:
        db = info.context.db
        from app.models.node import Node
        nodes = db.query(Node).filter(Node.project_id == self.id).all()
        from app.services.memory_estimator import MemoryEstimator
        return MemoryEstimator.estimate_project_metrics(nodes)["total_parameter_count"]

    @strawberry.field
    def total_parameter_memory_mb(self, info) -> float:
        db = info.context.db
        from app.models.node import Node
        nodes = db.query(Node).filter(Node.project_id == self.id).all()
        from app.services.memory_estimator import MemoryEstimator
        return MemoryEstimator.estimate_project_metrics(nodes)["total_parameter_memory_mb"]

    @strawberry.field
    def total_activation_memory_mb(self, info) -> float:
        db = info.context.db
        from app.models.node import Node
        nodes = db.query(Node).filter(Node.project_id == self.id).all()
        from app.services.memory_estimator import MemoryEstimator
        return MemoryEstimator.estimate_project_metrics(nodes)["total_activation_memory_mb"]

    @strawberry.field
    def total_flops(self, info) -> float:
        db = info.context.db
        from app.models.node import Node
        nodes = db.query(Node).filter(Node.project_id == self.id).all()
        from app.services.memory_estimator import MemoryEstimator
        return MemoryEstimator.estimate_project_metrics(nodes)["total_flops"]

    @strawberry.field
    def estimated_gpu_memory_mb(self, info) -> float:
        db = info.context.db
        from app.models.node import Node
        nodes = db.query(Node).filter(Node.project_id == self.id).all()
        from app.services.memory_estimator import MemoryEstimator
        return MemoryEstimator.estimate_project_metrics(nodes)["estimated_gpu_memory_mb"]

    @strawberry.field
    def execution_plan(self, info) -> ExecutionPlanType:
        db = info.context.db
        from app.models.node import Node
        from app.models.edge import Edge
        from app.models.project import Project
        
        # Load all nodes & edges
        db_nodes = db.query(Node).filter(Node.project_id == self.id).all()
        db_edges = db.query(Edge).filter(Edge.project_id == self.id).all()
        
        # Build IRGraph
        from app.ir.ir_graph import IRGraph
        project = db.query(Project).filter(Project.id == self.id).first()
        ir_graph = IRGraph.from_db(project, db_nodes, db_edges)
        
        # Generate execution plan
        from app.services.graph_engine import ExecutionPlanner
        plan_dict = ExecutionPlanner.get_execution_plan(ir_graph)
        
        from app.graphql.types.execution_plan_type import ExecutionPlanType, ExecutionStepType
        steps_list = [
            ExecutionStepType(
                step_index=s["step_index"],
                node_ids=s["node_ids"],
                dependencies=s["dependencies"]
            ) for s in plan_dict["steps"]
        ]
        return ExecutionPlanType(
            steps=steps_list,
            concurrency_limit=plan_dict["concurrency_limit"],
            total_steps=plan_dict["total_steps"]
        )
