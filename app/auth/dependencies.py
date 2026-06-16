from fastapi import Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from strawberry.fastapi import BaseContext
import uuid
import logging

from app.config.database import get_db
from app.auth.security import decode_access_token
from app.models.user import User
from app.config.settings import settings

logger = logging.getLogger("mlbuilder.auth")

security_scheme = HTTPBearer(auto_error=False)

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
    db: Session = Depends(get_db)
) -> User | None:
    """FastAPI dependency to retrieve the current user from the Bearer token.
    Automatically authenticates a default developer user in local development mode.
    """
    if not credentials:
        # Auto-authenticate default developer user in local development mode
        dev_user = db.query(User).filter(User.username == "developer").first()
        if not dev_user:
            dev_user = User(
                id=uuid.uuid4(),
                email="developer@mlbuilder.local",
                username="developer",
                password_hash="dev_hash",
                role="admin"  # Admin role gives full developer capabilities
            )
            db.add(dev_user)
            db.commit()
            db.refresh(dev_user)
        return dev_user
    
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        # In development mode, treat an invalid/expired token as if no token was
        # sent — fall through to the dev-user auto-login so the frontend's stale
        # tokens never block local iteration.
        if settings.ENVIRONMENT == "development":
            logger.warning(
                "JWT decode failed in development mode — falling back to dev user. "
                "This would return 401 in production."
            )
            dev_user = db.query(User).filter(User.username == "developer").first()
            if not dev_user:
                dev_user = User(
                    id=uuid.uuid4(),
                    email="developer@mlbuilder.local",
                    username="developer",
                    password_hash="dev_hash",
                    role="admin",
                )
                db.add(dev_user)
                db.commit()
                db.refresh(dev_user)
            return dev_user
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
