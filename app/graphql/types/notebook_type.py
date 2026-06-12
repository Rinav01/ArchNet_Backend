import strawberry

@strawberry.type
class NotebookCellResultType:
    success: bool
    stdout: str
    stderr: str
    execution_time_ms: int
