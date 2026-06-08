import strawberry

@strawberry.type
class ArchitectureTemplateType:
    name: str
    description: str
    nodes: strawberry.scalars.JSON
    edges: strawberry.scalars.JSON
