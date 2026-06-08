import strawberry

@strawberry.type
class ArchitectureScoreType:
    score: int
    grade: str
    breakdown: strawberry.scalars.JSON
