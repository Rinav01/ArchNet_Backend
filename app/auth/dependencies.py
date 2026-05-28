from fastapi import Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from strawberry.fastapi import BaseContext
import uuid

from app.config.database import get_db
from app.auth.security import decode_access_token
from app.models.user import User

security_scheme = HTTPBearer(auto_error=False)

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
    db: Session = Depends(get_db)
) -> User | None:
    """FastAPI dependency to retrieve the current user from the Bearer token."""
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        return None
    
    user_id_str = payload.get("sub")
    if not user_id_str:
        return None
    
    try:
        user_uuid = uuid.UUID(user_id_str)
    except ValueError:
        return None
        
    user = db.query(User).filter(User.id == user_uuid).first()
    return user

class GraphQLContext(BaseContext):
    """Custom context class for Strawberry GraphQL holding db, current user, and client IP."""
    def __init__(self, db: Session, current_user: User | None, ip_address: str | None = None):
        super().__init__()
        self.db = db
        self.current_user = current_user
        self.ip_address = ip_address

def get_graphql_context(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user)
) -> GraphQLContext:
    """FastAPI dependency to assemble the Strawberry GraphQL execution context."""
    ip_address = request.client.host if request and request.client else None
    return GraphQLContext(db=db, current_user=current_user, ip_address=ip_address)
