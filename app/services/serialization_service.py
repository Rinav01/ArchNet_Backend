import json
import uuid
from sqlalchemy.orm import Session
from app.models.project import Project
from app.models.node import Node
from app.models.edge import Edge
from app.ir.ir_graph import IRGraph
from app.services.project_service import ProjectService
from app.services.validation_service import ValidationService
from app.services.shape_inference_service import ShapeInferenceService

class SerializationService:
    @staticmethod
    def export_project(db: Session, project_id: uuid.UUID, user_id: uuid.UUID) -> str:
        """Serializes a database canvas project into a highly portable JSON schema string (based on IRGraph)."""
        project = ProjectService.get_project(db, project_id, user_id=user_id)
        if not project:
            raise ValueError("Project not found.")

        # Fetch active nodes and edges
        nodes = db.query(Node).filter(Node.project_id == project_id).all()
        edges = db.query(Edge).filter(Edge.project_id == project_id).all()

        # Construct framework-agnostic IRGraph
        ir_graph = IRGraph.from_db(project, nodes, edges)
        
        # Serialize to standard JSON Graph Export format
        return json.dumps(ir_graph.to_dict(), indent=2)

    @staticmethod
    def import_project(db: Session, user_id: uuid.UUID, name: str, graph_data_str: str) -> Project:
        """Parses a serialized JSON graph payload, creates a new database canvas project, and rebuilds nodes/edges.
        Performs full graph validation and shape inference on import.
        """
        try:
            data = json.loads(graph_data_str)
            ir_graph = IRGraph.from_dict(data)
        except Exception as e:
            raise ValueError(f"Invalid graph export schema or parsing error: {e}")

        # 1. Create a new database project workspace
        new_project = ProjectService.create_project(
            db=db,
            user_id=user_id,
            name=name.strip(),
            description=f"Imported from {ir_graph.project_name}",
            framework=ir_graph.framework
        )

        # 2. Re-create all nodes, maintaining connection mappings
        old_id_to_new_id: dict[str, uuid.UUID] = {}
        
        for old_id, ir_node in ir_graph.nodes.items():
            new_node = ProjectService.add_node(
                db=db,
                project_id=new_project.id,
                node_type=ir_node.op_type,
                label=ir_node.label,
                position_x=ir_node.params.get("position_x", 0.0),
                position_y=ir_node.params.get("position_y", 0.0),
                config=ir_node.params
            )
            old_id_to_new_id[old_id] = new_node.id

        # 3. Re-create all connecting edges using the mapping index
        for old_id, ir_node in ir_graph.nodes.items():
            new_to_id = old_id_to_new_id[old_id]
            for old_parent_id in ir_node.inputs:
                if old_parent_id in old_id_to_new_id:
                    new_from_id = old_id_to_new_id[old_parent_id]
                    ProjectService.add_edge(
                        db=db,
                        project_id=new_project.id,
                        from_node_id=new_from_id,
                        to_node_id=new_to_id
                    )

        # 4. Trigger validation and shape compilation on the imported graph immediately
        try:
            nodes_list = db.query(Node).filter(Node.project_id == new_project.id).all()
            edges_list = db.query(Edge).filter(Edge.project_id == new_project.id).all()
            
            sorted_nodes = ValidationService.validate_graph(nodes_list, edges_list)
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges_list)
            db.commit()
        except Exception as validation_err:
            # Commit the base project structural import but log shape validation details
            db.rollback()
            raise ValueError(f"Import warning: Graph structure imported successfully but failed shape checks: {validation_err}")

        db.refresh(new_project)
        return new_project
