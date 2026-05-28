import uuid
from sqlalchemy.orm import Session
from app.models.user import User
from app.models.project import Project
from app.models.dataset import Dataset

def verify_role_and_ownership(
    info,
    allowed_roles: list[str],
    project_id: uuid.UUID | str | None = None,
    dataset_id: uuid.UUID | str | None = None
) -> User:
    """Verifies that the current user holds one of the allowed_roles,
    and has correct ownership access to target projects or datasets (unless admin).
    """
    user = info.context.current_user
    if not user:
        raise Exception("Not authenticated.")
        
    db = info.context.db
    
    # 1. Enforce Global Role authorization
    if user.role not in allowed_roles:
        raise Exception("Forbidden: Insufficient permissions.")
        
    # 2. Admins bypass resource ownership filters
    if user.role == "admin":
        return user
        
    # 3. Enforce Project resource checks
    if project_id:
        proj_uuid = uuid.UUID(str(project_id)) if not isinstance(project_id, uuid.UUID) else project_id
        project = db.query(Project).filter(Project.id == proj_uuid).first()
        if not project:
            raise Exception("Project not found.")
        if project.user_id != user.id:
            # Check if this is a query and the project is public
            is_query = info.field_name in {"project", "projects", "export_project"}
            if not (is_query and project.is_public):
                raise Exception("Forbidden: You do not own this project.")
                
    # 4. Enforce Dataset resource checks
    if dataset_id:
        ds_uuid = uuid.UUID(str(dataset_id)) if not isinstance(dataset_id, uuid.UUID) else dataset_id
        dataset = db.query(Dataset).filter(Dataset.id == ds_uuid).first()
        if not dataset:
            raise Exception("Dataset not found.")
        if dataset.user_id != user.id:
            raise Exception("Forbidden: You do not own this dataset.")
            
    return user
