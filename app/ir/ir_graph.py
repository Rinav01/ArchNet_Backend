from pydantic import BaseModel, Field
from typing import Dict, Any, List
from app.ir.ir_node import IRNode

class IRGraph(BaseModel):
    """Framework-agnostic intermediate representation of an entire neural network model architecture graph."""
    project_id: str
    project_name: str
    framework: str
    nodes: Dict[str, IRNode] = Field(default_factory=dict)

    def add_node(self, node: IRNode) -> None:
        self.nodes[node.id] = node

    @classmethod
    def from_db(cls, project: Any, db_nodes: List[Any], db_edges: List[Any]) -> "IRGraph":
        """Factory method to construct an IRGraph from SQLAlchemy database models."""
        graph = cls(
            project_id=str(project.id),
            project_name=project.name,
            framework=project.framework
        )

        # 1. Initialize all nodes
        for db_node in db_nodes:
            ir_node = IRNode(
                id=str(db_node.id),
                op_type=db_node.type,
                label=db_node.label,
                params=db_node.config or {},
                input_shape=db_node.input_shape,
                output_shape=db_node.output_shape
            )
            graph.add_node(ir_node)

        # 2. Build adjacency connections (inputs & outputs lists) from edges
        for db_edge in db_edges:
            from_id = str(db_edge.from_node_id)
            to_id = str(db_edge.to_node_id)
            
            if from_id in graph.nodes and to_id in graph.nodes:
                graph.nodes[from_id].add_output(to_id)
                graph.nodes[to_id].add_input(from_id)

        return graph

    def get_topologically_sorted_nodes(self) -> List[IRNode]:
        """Perform a topological sort on the active IRNodes inside the graph.
        Returns a sorted list of IRNode instances. Raises ValueError if a cycle exists.
        """
        in_degree = {nid: len(node.inputs) for nid, node in self.nodes.items()}
        
        # Queue of nodes with no incoming connections (sources)
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_nodes: List[IRNode] = []

        while queue:
            u = queue.pop(0)
            node = self.nodes[u]
            sorted_nodes.append(node)

            for v in node.outputs:
                if v in in_degree:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Invalid Graph: Contain cyclic dependencies.")

        return sorted_nodes

    def to_dict(self) -> Dict[str, Any]:
        """Convert the IRGraph to a standard dictionary format for JSON serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IRGraph":
        """Reconstruct the IRGraph from a standard dictionary format."""
        return cls.model_validate(data)
