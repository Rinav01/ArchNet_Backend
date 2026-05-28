from typing import List, Dict, Set, Any
from app.ir.ir_graph import IRGraph
from app.ir.ir_node import IRNode

class ExecutionPlanner:
    @staticmethod
    def get_execution_plan(ir_graph: IRGraph) -> List[str]:
        """Generates a scheduled list of Node IDs in topological order ready for execution mapping."""
        sorted_nodes = ir_graph.get_topologically_sorted_nodes()
        return [node.id for node in sorted_nodes]

class GraphOptimizer:
    @staticmethod
    def prune_dead_nodes(ir_graph: IRGraph, output_node_id: str | None = None) -> int:
        """Prunes 'dead' nodes that do not reach the final terminating output node of the network.
        Defines the final terminating node as the last node in the topological execution order,
        or uses the explicitly specified output_node_id.
        Returns the number of pruned nodes.
        """
        nodes = ir_graph.nodes
        if not nodes:
            return 0

        # 1. Identify active output sink
        if output_node_id and output_node_id in nodes:
            final_output_id = output_node_id
        else:
            try:
                sorted_nodes = ir_graph.get_topologically_sorted_nodes()
            except ValueError:
                return 0  # Cyclic graph: skip pruning to prevent corrupting validation errors
                
            if not sorted_nodes:
                return 0
            final_output_id = sorted_nodes[-1].id

        # 2. Perform backward search (BFS) starting from the final output node along inputs
        visited: Set[str] = set()
        queue = [final_output_id]

        while queue:
            curr_id = queue.pop(0)
            if curr_id not in visited:
                visited.add(curr_id)
                curr_node = nodes.get(curr_id)
                if curr_node:
                    for parent_id in curr_node.inputs:
                        if parent_id in nodes and parent_id not in visited:
                            queue.append(parent_id)

        # 3. Find dead nodes (nodes that cannot reach the final output)
        dead_node_ids = set(nodes.keys()) - visited
        
        # 4. Disconnect and remove dead nodes
        for dead_id in dead_node_ids:
            dead_node = nodes[dead_id]
            
            # Remove reference from parent outputs
            for parent_id in dead_node.inputs:
                if parent_id in nodes:
                    if dead_id in nodes[parent_id].outputs:
                        nodes[parent_id].outputs.remove(dead_id)
            
            # Remove reference from child inputs
            for child_id in dead_node.outputs:
                if child_id in nodes:
                    if dead_id in nodes[child_id].inputs:
                        nodes[child_id].inputs.remove(dead_id)

            del nodes[dead_id]

        return len(dead_node_ids)

    @staticmethod
    def simplify_graph(ir_graph: IRGraph) -> None:
        """Run all graph optimization passes: dead node pruning, connection simplifications, etc."""
        # Run dead node pruning
        GraphOptimizer.prune_dead_nodes(ir_graph)
