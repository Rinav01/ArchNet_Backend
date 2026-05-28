from pydantic import BaseModel, Field
from typing import List, Dict, Any

class IRNode(BaseModel):
    """Framework-agnostic intermediate representation of a single neural network layer/operation node."""
    id: str
    op_type: str  # e.g., 'Input', 'Conv2D', 'MaxPool2D', 'Flatten', 'Dense', 'Add', 'Concatenate'
    label: str
    inputs: List[str] = Field(default_factory=list)  # Predecessor node IDs
    outputs: List[str] = Field(default_factory=list)  # Successor node IDs
    params: Dict[str, Any] = Field(default_factory=dict)  # Layer configurations (filters, kernel_size, etc.)
    input_shape: Any = None  # List[int] for single-input, Dict[str, List[int]] for multi-input
    output_shape: List[Any] | None = None

    def add_input(self, node_id: str) -> None:
        if node_id not in self.inputs:
            self.inputs.append(node_id)

    def add_output(self, node_id: str) -> None:
        if node_id not in self.outputs:
            self.outputs.append(node_id)
