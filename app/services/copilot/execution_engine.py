import uuid
import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.models.node import Node
from app.models.edge import Edge
from app.models.project import Project
from app.ir.ir_graph import IRGraph
from app.services.shape_inference_service import ShapeInferenceService
from app.services.caching_service import CachingService
from app.services.event_dispatcher import EventDispatcher

logger = logging.getLogger("mlbuilder.copilot.execution_engine")

class CopilotExecutionEngine:
    @staticmethod
    def execute_graph_replacement(
        db: Session,
        project_id: uuid.UUID,
        generated_graph: Dict[str, Any]
    ) -> IRGraph:
        """Deletes all existing nodes and edges for the project and replaces them
        with the newly generated graph. Applies topological layout and runs shape inference.
        """
        # 1. Verify project exists
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project with ID {project_id} not found.")

        # 2. Delete all existing edges and nodes
        db.query(Edge).filter(Edge.project_id == project_id).delete()
        db.query(Node).filter(Node.project_id == project_id).delete()
        db.flush()

        # 3. Calculate topological layout positions for the new nodes
        nodes_raw = generated_graph.get("nodes", [])
        edges_raw = generated_graph.get("edges", [])

        # Assign unique UUIDs and calculate positions
        id_map: Dict[str, uuid.UUID] = {}
        nodes_by_id = {n["id"]: n for n in nodes_raw}
        
        # Build connections for layout calculation
        adjacency: Dict[str, List[str]] = {n["id"]: [] for n in nodes_raw}
        in_degree: Dict[str, int] = {n["id"]: 0 for n in nodes_raw}
        
        for edge in edges_raw:
            src = edge["from_node_id"]
            dst = edge["to_node_id"]
            if src in adjacency and dst in adjacency:
                adjacency[src].append(dst)
                in_degree[dst] += 1

        # Level-by-level BFS to assign vertical layers (topological order)
        levels: Dict[str, int] = {}
        sources = [nid for nid, deg in in_degree.items() if deg == 0]
        
        # In case there's a cycle or no sources, put everything at level 0
        if not sources and nodes_raw:
            for n in nodes_raw:
                levels[n["id"]] = 0
        else:
            queue = [(nid, 0) for nid in sources]
            while queue:
                curr_id, lvl = queue.pop(0)
                levels[curr_id] = max(levels.get(curr_id, 0), lvl)
                for neighbor in adjacency[curr_id]:
                    # Update neighbor level and append to queue
                    levels[neighbor] = max(levels.get(neighbor, 0), lvl + 1)
                    queue.append((neighbor, lvl + 1))

        # Group node IDs by level
        level_groups: Dict[int, List[str]] = {}
        for nid, lvl in levels.items():
            if lvl not in level_groups:
                level_groups[lvl] = []
            level_groups[lvl].append(nid)

        # 4. Create Node models with computed positions
        db_nodes: Dict[str, Node] = {}
        for lvl, nids in level_groups.items():
            num_nodes = len(nids)
            for idx, nid in enumerate(nids):
                raw_node = nodes_by_id[nid]
                new_id = uuid.uuid4()
                id_map[nid] = new_id

                # Horizontal centering layout
                pos_x = (idx - (num_nodes - 1) / 2.0) * 250.0 + 300.0
                pos_y = lvl * 180.0 + 100.0

                node_model = Node(
                    id=new_id,
                    project_id=project_id,
                    type=raw_node["type"],
                    label=raw_node["label"],
                    position_x=pos_x,
                    position_y=pos_y,
                    config=raw_node.get("config", {})
                )
                db.add(node_model)
                db_nodes[nid] = node_model

        # If any node was left out due to disconnect/cycles, add it at level 0
        for raw_node in nodes_raw:
            if raw_node["id"] not in id_map:
                new_id = uuid.uuid4()
                id_map[raw_node["id"]] = new_id
                node_model = Node(
                    id=new_id,
                    project_id=project_id,
                    type=raw_node["type"],
                    label=raw_node["label"],
                    position_x=300.0,
                    position_y=100.0,
                    config=raw_node.get("config", {})
                )
                db.add(node_model)
                db_nodes[raw_node["id"]] = node_model

        db.flush()

        # 5. Create Edge models
        db_edges: List[Edge] = []
        for edge in edges_raw:
            src = edge["from_node_id"]
            dst = edge["to_node_id"]
            if src in id_map and dst in id_map:
                edge_model = Edge(
                    id=uuid.uuid4(),
                    project_id=project_id,
                    from_node_id=id_map[src],
                    to_node_id=id_map[dst]
                )
                db.add(edge_model)
                db_edges.append(edge_model)

        db.flush()

        # 6. Run shape inference
        nodes_list = list(db_nodes.values())
        ir_graph = IRGraph.from_db(project, nodes_list, db_edges)
        try:
            sorted_ir_nodes = ir_graph.get_topologically_sorted_nodes()
            model_by_id = {str(n.id): n for n in nodes_list}
            sorted_models = [model_by_id[str(n.id)] for n in sorted_ir_nodes]
        except Exception:
            sorted_models = nodes_list

        try:
            ShapeInferenceService.run_shape_inference(sorted_models, db_edges)
        except Exception as e:
            logger.warning(f"Shape inference failed during graph replacement execution: {e}")

        # Commit all changes to DB
        db.commit()

        # Invalidate project cache
        CachingService.invalidate_project_cache(project_id)

        # Dispatch node added event for the new nodes
        for node in nodes_list:
            try:
                EventDispatcher.dispatch_node_added(project_id, node.id, node.label, node.type)
            except Exception:
                pass

        # Re-fetch from DB to return accurate, complete IRGraph
        db.refresh(project)
        db_nodes_final = db.query(Node).filter(Node.project_id == project_id).all()
        db_edges_final = db.query(Edge).filter(Edge.project_id == project_id).all()
        return IRGraph.from_db(project, db_nodes_final, db_edges_final)
