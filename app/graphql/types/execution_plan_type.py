import strawberry
from typing import List

@strawberry.type
class ExecutionStepType:
    step_index: int
    node_ids: List[str]
    dependencies: List[str]

@strawberry.type
class ExecutionPlanType:
    steps: List[ExecutionStepType]
    concurrency_limit: int
    total_steps: int
