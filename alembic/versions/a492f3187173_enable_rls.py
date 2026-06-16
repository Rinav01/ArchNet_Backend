"""enable_rls

Revision ID: a492f3187173
Revises: cc677ea9daeb
Create Date: 2026-06-13 00:57:24.527727

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a492f3187173'
down_revision: Union[str, None] = 'cc677ea9daeb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable RLS on all tables
    tables = [
        "users", "projects", "nodes", "edges", "datasets", "dataset_versions", 
        "experiments", "model_versions", "models", "model_artifacts", 
        "export_artifacts", "training_jobs", "training_runs", "deployments", 
        "deployment_metrics", "audit_logs", "workflows", "workflow_runs"
    ]
    for table in tables:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
        
    # Note: Table owners and superusers bypass RLS automatically unless FORCE is used,
    # so the backend connection (running as database owner) will bypass RLS by default.
    pass


def downgrade() -> None:
    tables = [
        "users", "projects", "nodes", "edges", "datasets", "dataset_versions", 
        "experiments", "model_versions", "models", "model_artifacts", 
        "export_artifacts", "training_jobs", "training_runs", "deployments", 
        "deployment_metrics", "audit_logs", "workflows", "workflow_runs"
    ]
    for table in tables:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")

