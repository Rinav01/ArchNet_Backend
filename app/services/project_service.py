from sqlalchemy.orm import Session
import uuid
from app.models.project import Project
from app.models.node import Node
from app.models.edge import Edge

class ProjectService:
    @staticmethod
    def create_project(
        db: Session, 
        user_id: uuid.UUID, 
        name: str, 
        description: str | None = None, 
        framework: str = "PyTorch"
    ) -> Project:
        """Create a new project for a user."""
        new_project = Project(
            user_id=user_id,
            name=name.strip(),
            description=description.strip() if description else None,
            framework=framework,
            is_public=False
        )
        db.add(new_project)
        db.commit()
        db.refresh(new_project)
        return new_project

    @staticmethod
    def get_project(db: Session, project_id: uuid.UUID, user_id: uuid.UUID | None = None) -> Project | None:
        """Retrieve a project by ID, optionally verifying ownership or public access."""
        query = db.query(Project).filter(Project.id == project_id)
        project = query.first()
        if not project:
            return None
        
        # If user_id is provided, verify ownership or public access
        if user_id and project.user_id != user_id and not project.is_public:
            raise PermissionError("You do not have permission to access this project.")
            
        return project

    @staticmethod
    def list_projects(db: Session, user_id: uuid.UUID, limit: int = 20, offset: int = 0) -> list[Project]:
        """List all projects belonging to a user."""
        return db.query(Project)\
            .filter(Project.user_id == user_id)\
            .order_by(Project.created_at.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()

    @staticmethod
    def add_node(
        db: Session, 
        project_id: uuid.UUID, 
        node_type: str, 
        label: str, 
        position_x: float, 
        position_y: float, 
        config: dict
    ) -> Node:
        """Add a new canvas node (layer) to a project."""
        new_node = Node(
            project_id=project_id,
            type=node_type,
            label=label,
            position_x=position_x,
            position_y=position_y,
            config=config,
            input_shape=None,
            output_shape=None
        )
        db.add(new_node)
        db.commit()
        db.refresh(new_node)
        return new_node

    @staticmethod
    def add_edge(
        db: Session, 
        project_id: uuid.UUID, 
        from_node_id: uuid.UUID, 
        to_node_id: uuid.UUID
    ) -> Edge:
        """Add a directed edge connecting two canvas nodes."""
        # Verify both nodes exist and belong to the same project
        from_node = db.query(Node).filter(Node.id == from_node_id, Node.project_id == project_id).first()
        to_node = db.query(Node).filter(Node.id == to_node_id, Node.project_id == project_id).first()
        
        if not from_node or not to_node:
            raise ValueError("Both source and destination nodes must exist in this project.")
            
        # Avoid duplicate edges
        existing_edge = db.query(Edge).filter(
            Edge.project_id == project_id,
            Edge.from_node_id == from_node_id,
            Edge.to_node_id == to_node_id
        ).first()
        
        if existing_edge:
            return existing_edge

        new_edge = Edge(
            project_id=project_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id
        )
        db.add(new_edge)
        db.commit()
        db.refresh(new_edge)
        return new_edge

    @staticmethod
    def delete_node(db: Session, project_id: uuid.UUID, node_id: uuid.UUID) -> bool:
        """Delete a node from a project, cascadingly deleting attached edges."""
        node = db.query(Node).filter(Node.id == node_id, Node.project_id == project_id).first()
        if not node:
            return False
            
        db.delete(node)
        db.commit()
        return True

    @staticmethod
    def delete_edge(db: Session, project_id: uuid.UUID, edge_id: uuid.UUID) -> bool:
        """Delete a connecting edge from a project."""
        edge = db.query(Edge).filter(Edge.id == edge_id, Edge.project_id == project_id).first()
        if not edge:
            return False
            
        db.delete(edge)
        db.commit()
        return True
