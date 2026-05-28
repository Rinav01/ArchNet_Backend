import uuid
import logging
from sqlalchemy.orm import Session
from app.models.audit_log import AuditLog

logger = logging.getLogger("mlbuilder.audit_service")

class AuditService:
    @staticmethod
    def log_action(
        db: Session,
        user_id: uuid.UUID | None,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        details: dict | None = None,
        ip_address: str | None = None
    ) -> AuditLog:
        """Create and persist an enterprise audit log in the database."""
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address
        )
        db.add(audit_log)
        db.commit()
        logger.info(f"AUDIT LOG: user={user_id} action={action} resource={resource_type}:{resource_id} IP={ip_address}")
        return audit_log
