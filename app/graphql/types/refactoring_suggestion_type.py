import strawberry

@strawberry.type
class RefactorActionType:
    type: str  # Insert Pooling, Insert BatchNorm, Replace Layers, Remove Bottlenecks
    params: strawberry.scalars.JSON | None

@strawberry.type
class RefactoringSuggestionType:
    category: str  # Memory Optimization, Latency Optimization, Parameter Reduction
    description: str
    action: RefactorActionType | None
