import strawberry
import uuid
from datetime import datetime

@strawberry.type
class DeploymentMetricsType:
    id: uuid.UUID
    deployment_id: uuid.UUID
    timestamp: datetime
    requests_count: int
    latency_ms: float
    error_count: int
    memory_mb: float
    gpu_usage_pct: float
