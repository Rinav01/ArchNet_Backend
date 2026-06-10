from app.ir.ir_graph import IRGraph

class GraphContextBuilder:
    @staticmethod
    def get_graph_summary(ir_graph: IRGraph) -> str:
        """Converts the IRGraph structure into a clean, text-readable representation for LLM consumption."""
        if not ir_graph.nodes:
            return "The graph is currently empty."
            
        summary = []
        summary.append(f"Project ID: {ir_graph.project_id}")
        summary.append(f"Project Name: {ir_graph.project_name}")
        summary.append(f"Framework: {ir_graph.framework}")
        summary.append("Nodes (Layers):")
        
        # Sort topologically if possible, otherwise list all
        try:
            sorted_nodes = ir_graph.get_topologically_sorted_nodes()
        except Exception:
            sorted_nodes = list(ir_graph.nodes.values())
            
        for node in sorted_nodes:
            summary.append(
                f"- Node ID: {node.id}\n"
                f"  Type: {node.op_type}\n"
                f"  Label: {node.label}\n"
                f"  Params: {node.params}\n"
                f"  Input Shape: {node.input_shape}\n"
                f"  Output Shape: {node.output_shape}"
            )
            
        summary.append("Edges (Connections):")
        # List all connections
        has_edges = False
        for node in ir_graph.nodes.values():
            for output_id in node.outputs:
                if output_id in ir_graph.nodes:
                    summary.append(f"- {node.id} -> {output_id}")
                    has_edges = True
                    
        if not has_edges:
            summary.append("(No connections)")
            
        return "\n".join(summary)
