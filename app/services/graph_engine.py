from typing import List, Dict, Set, Any
from app.ir.ir_graph import IRGraph
from app.ir.ir_node import IRNode

class ExecutionPlanner:
    @staticmethod
    def get_execution_plan(ir_graph: IRGraph) -> Dict[str, Any]:
        """Generates a scheduled list of steps with level-by-level dependencies and concurrency analysis."""
        nodes = ir_graph.nodes
        if not nodes:
            return {
                "steps": [],
                "concurrency_limit": 0,
                "total_steps": 0
            }
        
        try:
            sorted_nodes = ir_graph.get_topologically_sorted_nodes()
        except ValueError:
            # Fallback if cyclic: return a flat single step to avoid crashing API
            return {
                "steps": [{
                    "step_index": 0,
                    "node_ids": list(nodes.keys()),
                    "dependencies": []
                }],
                "concurrency_limit": len(nodes),
                "total_steps": 1
            }
        
        node_levels: Dict[str, int] = {}
        
        for node in sorted_nodes:
            if not node.inputs:
                node_levels[node.id] = 0
            else:
                max_parent_level = 0
                for parent_id in node.inputs:
                    if parent_id in node_levels:
                        max_parent_level = max(max_parent_level, node_levels[parent_id])
                node_levels[node.id] = max_parent_level + 1
                
        # Group by level
        level_to_nodes: Dict[int, List[str]] = {}
        for nid, lvl in node_levels.items():
            if lvl not in level_to_nodes:
                level_to_nodes[lvl] = []
            level_to_nodes[lvl].append(nid)
            
        steps = []
        max_lvl = max(level_to_nodes.keys()) if level_to_nodes else -1
        concurrency_limit = 0
        
        for lvl in range(max_lvl + 1):
            if lvl not in level_to_nodes:
                continue
            node_ids = level_to_nodes[lvl]
            concurrency_limit = max(concurrency_limit, len(node_ids))
            
            # Find dependencies: all inputs of nodes in this step
            deps = set()
            for nid in node_ids:
                node_obj = nodes.get(nid)
                if node_obj:
                    for parent_id in node_obj.inputs:
                        if parent_id in nodes:
                            deps.add(parent_id)
                            
            steps.append({
                "step_index": lvl,
                "node_ids": node_ids,
                "dependencies": list(deps)
            })
            
        return {
            "steps": steps,
            "concurrency_limit": concurrency_limit,
            "total_steps": len(steps)
        }

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
    def eliminate_identity_nodes(ir_graph: IRGraph) -> int:
        """Removes any identity nodes, reconnecting their inputs directly to their outputs."""
        nodes = ir_graph.nodes
        count = 0
        while True:
            identity_id = None
            for nid, node in nodes.items():
                if node.op_type.lower() in ("identity", "identity_node"):
                    identity_id = nid
                    break
            if not identity_id:
                break
                
            identity_node = nodes[identity_id]
            parents = list(identity_node.inputs)
            children = list(identity_node.outputs)
            
            # Reconnect parents to children
            for parent_id in parents:
                if parent_id in nodes:
                    parent_node = nodes[parent_id]
                    if identity_id in parent_node.outputs:
                        parent_node.outputs.remove(identity_id)
                    for child_id in children:
                        if child_id not in parent_node.outputs:
                            parent_node.outputs.append(child_id)
                            
            # Reconnect children to parents
            for child_id in children:
                if child_id in nodes:
                    child_node = nodes[child_id]
                    if identity_id in child_node.inputs:
                        child_node.inputs.remove(identity_id)
                    for parent_id in parents:
                        if parent_id not in child_node.inputs:
                            child_node.inputs.append(parent_id)
                            
            del nodes[identity_id]
            count += 1
        return count

    @staticmethod
    def fuse_conv_bn_layers(ir_graph: IRGraph) -> int:
        """Fuses Conv2D -> BatchNorm2D sequences.
        If Conv2D output has exactly 1 output (which is BatchNorm2D) and BatchNorm2D has exactly 1 input (Conv2D),
        we merge BN configuration into Conv2D (set fused_batchnorm=True) and prune BatchNorm2D.
        """
        nodes = ir_graph.nodes
        fused_count = 0
        
        while True:
            fuse_pair = None
            for nid, node in nodes.items():
                if node.op_type.lower() == "conv2d" and len(node.outputs) == 1:
                    child_id = node.outputs[0]
                    child_node = nodes.get(child_id)
                    if child_node and child_node.op_type.lower() in ("batchnorm", "batchnorm2d") and len(child_node.inputs) == 1:
                        fuse_pair = (nid, child_id)
                        break
            
            if not fuse_pair:
                break
                
            conv_id, bn_id = fuse_pair
            conv_node = nodes[conv_id]
            bn_node = nodes[bn_id]
            
            # Mark Conv2D node as having fused batchnorm
            if not conv_node.params:
                conv_node.params = {}
            conv_node.params["fused_batchnorm"] = True
            
            # Bypass BN node: reconnect Conv2D to BN outputs
            bn_children = list(bn_node.outputs)
            conv_node.outputs = bn_children
            
            # For each child of BN, replace BN with Conv2D in its inputs
            for child_id in bn_children:
                if child_id in nodes:
                    child_node = nodes[child_id]
                    if bn_id in child_node.inputs:
                        child_node.inputs.remove(bn_id)
                    if conv_id not in child_node.inputs:
                        child_node.inputs.append(conv_id)
                        
            # Delete BN node
            del nodes[bn_id]
            fused_count += 1
            
        return fused_count

    @staticmethod
    def simplify_graph(ir_graph: IRGraph) -> None:
        """Run all graph optimization passes: dead node pruning, connection simplifications, etc."""
        GraphOptimizer.prune_dead_nodes(ir_graph)
        GraphOptimizer.eliminate_identity_nodes(ir_graph)
        GraphOptimizer.fuse_conv_bn_layers(ir_graph)
