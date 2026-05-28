from abc import ABC, abstractmethod
from app.ir.ir_graph import IRGraph

class BaseCompiler(ABC):
    """Abstract Base Compiler class establishing standard interface for framework code-generators."""
    
    @abstractmethod
    def compile(self, ir_graph: IRGraph) -> str:
        """Compile a framework-agnostic IRGraph into a runnable framework-specific Python script."""
        pass
