import uuid
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.node import Node
from app.models.edge import Edge
from app.services.event_dispatcher import EventDispatcher
from app.services.caching_service import CachingService

logger = logging.getLogger("mlbuilder.crdt_resolver")

class CRDTOperationResolver:
    # Track recently deleted items (Tombstones) to resolve out-of-order latency updates
    _tombstones = set()

    @classmethod
    def apply_operation(cls, db: Session, project_id: uuid.UUID, op: dict) -> dict:
        """Processes and applies a collaborative canvas operational message
        following strict Conflict-Free Replicated Data Type (CRDT) LWW & tombstoning rules.
        """
        action = op.get("action")
        payload = op.get("payload", {})
        client_timestamp = op.get("timestamp")
        
        if not client_timestamp:
            client_timestamp = datetime.now().timestamp()
            
        op_datetime = datetime.fromtimestamp(client_timestamp)


        # 1. ADD_NODE
        if action == "ADD_NODE":
            node_id_str = payload.get("node_id")
            node_uuid = uuid.UUID(node_id_str) if node_id_str else uuid.uuid4()
            
            if node_uuid in cls._tombstones:
                logger.warning(f"Discard ADD_NODE: Node {node_uuid} has been deleted.")
                return {"success": False, "reason": "Node has been deleted."}

            existing = db.query(Node).filter(Node.id == node_uuid).first()
            if existing:
                return {"success": True, "node_id": str(existing.id), "status": "exists"}

            new_node = Node(
                id=node_uuid,
                project_id=project_id,
                type=payload.get("type", "Dense"),
                label=payload.get("label", "Layer"),
                position_x=float(payload.get("position_x", 0.0)),
                position_y=float(payload.get("position_y", 0.0)),
                config=payload.get("config", {}),
                created_at=op_datetime,
                updated_at=op_datetime
            )
            db.add(new_node)
            db.commit()
            db.refresh(new_node)
            CachingService.invalidate_project_cache(project_id)
            
            # Dispatch NodeAdded event globally
            EventDispatcher.dispatch_node_added(project_id, new_node.id, new_node.label, new_node.type)
            return {"success": True, "action": action, "node": {
                "id": str(new_node.id),
                "type": new_node.type,
                "label": new_node.label,
                "position_x": new_node.position_x,
                "position_y": new_node.position_y,
                "config": new_node.config
            }}

        # 2. MOVE_NODE
        elif action == "MOVE_NODE":
            node_uuid = uuid.UUID(payload["node_id"])
            if node_uuid in cls._tombstones:
                return {"success": False, "reason": "Node has been deleted."}

            node = db.query(Node).filter(Node.id == node_uuid, Node.project_id == project_id).first()
            if not node:
                return {"success": False, "reason": "Node not found."}

            # LWW (Last-Write-Wins) timestamp check
            if node.updated_at is None or op_datetime > node.updated_at:
                node.position_x = float(payload["position_x"])
                node.position_y = float(payload["position_y"])
                node.updated_at = op_datetime
                db.commit()
                CachingService.invalidate_project_cache(project_id)
                return {"success": True, "action": action, "payload": payload}
            else:
                logger.info(f"Ignored out-of-order node move: {node_uuid}")
                return {"success": False, "reason": "Ignored out-of-order operation (older timestamp)."}

        # 3. UPDATE_NODE_CONFIG
        elif action == "UPDATE_NODE_CONFIG":
            node_uuid = uuid.UUID(payload["node_id"])
            if node_uuid in cls._tombstones:
                return {"success": False, "reason": "Node has been deleted."}

            node = db.query(Node).filter(Node.id == node_uuid, Node.project_id == project_id).first()
            if not node:
                return {"success": False, "reason": "Node not found."}

            # LWW configuration update check
            if node.updated_at is None or op_datetime > node.updated_at:
                current_config = node.config or {}
                incoming_config = payload.get("config", {})
                
                # Merge config fields selectively
                merged_config = {**current_config, **incoming_config}
                node.config = merged_config
                node.updated_at = op_datetime
                db.commit()
                CachingService.invalidate_project_cache(project_id)
                return {"success": True, "action": action, "payload": {
                    "node_id": str(node_uuid),
                    "config": merged_config
                }}
            else:
                return {"success": False, "reason": "Ignored out-of-order operation (older timestamp)."}

        # 4. DELETE_NODE
        elif action == "DELETE_NODE":
            node_uuid = uuid.UUID(payload["node_id"])
            cls._tombstones.add(node_uuid)

            node = db.query(Node).filter(Node.id == node_uuid, Node.project_id == project_id).first()
            if not node:
                # Discard cleanly if already deleted
                return {"success": True, "action": action, "status": "already_deleted"}

            # Deletion cascades to connected edges
            attached_edges = db.query(Edge).filter(
                (Edge.from_node_id == node_uuid) | (Edge.to_node_id == node_uuid)
            ).all()
            for edge in attached_edges:
                db.delete(edge)

            db.delete(node)
            db.commit()
            CachingService.invalidate_project_cache(project_id)

            # Dispatch NodeDeleted event globally
            EventDispatcher.dispatch_node_deleted(project_id, node_uuid)
            return {"success": True, "action": action, "payload": payload}

        # 5. ADD_EDGE
        elif action == "ADD_EDGE":
            from_node_uuid = uuid.UUID(payload["from_node_id"])
            to_node_uuid = uuid.UUID(payload["to_node_id"])
            
            # Semantic Guardrail: Assert both nodes are active and NOT concurrently deleted
            if from_node_uuid in cls._tombstones or to_node_uuid in cls._tombstones:
                return {"success": False, "reason": "Cannot link edge to a deleted node."}

            from_exists = db.query(Node).filter(Node.id == from_node_uuid, Node.project_id == project_id).first()
            to_exists = db.query(Node).filter(Node.id == to_node_uuid, Node.project_id == project_id).first()
            
            if not from_exists or not to_exists:
                return {"success": False, "reason": "Both layer endpoints must actively exist to form an edge."}

            edge_uuid_str = payload.get("edge_id")
            edge_uuid = uuid.UUID(edge_uuid_str) if edge_uuid_str else uuid.uuid4()

            existing_edge = db.query(Edge).filter(
                (Edge.project_id == project_id) &
                (Edge.from_node_id == from_node_uuid) &
                (Edge.to_node_id == to_node_uuid)
            ).first()
            
            if existing_edge:
                return {"success": True, "edge_id": str(existing_edge.id), "status": "exists"}

            new_edge = Edge(
                id=edge_uuid,
                project_id=project_id,
                from_node_id=from_node_uuid,
                to_node_id=to_node_uuid,
                created_at=op_datetime
            )
            db.add(new_edge)
            db.commit()
            db.refresh(new_edge)
            CachingService.invalidate_project_cache(project_id)
            
            return {"success": True, "action": action, "edge": {
                "id": str(new_edge.id),
                "from_node_id": str(new_edge.from_node_id),
                "to_node_id": str(new_edge.to_node_id)
            }}

        # 6. DELETE_EDGE
        elif action == "DELETE_EDGE":
            edge_uuid = uuid.UUID(payload["edge_id"])
            edge = db.query(Edge).filter(Edge.id == edge_uuid, Edge.project_id == project_id).first()
            
            if not edge:
                return {"success": True, "action": action, "status": "already_deleted"}

            db.delete(edge)
            db.commit()
            CachingService.invalidate_project_cache(project_id)
            return {"success": True, "action": action, "payload": payload}

        else:
            return {"success": False, "reason": f"Unknown operation action: {action}"}
