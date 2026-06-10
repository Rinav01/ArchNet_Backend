import os
import re
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader

from app.codegen.base_compiler import BaseCompiler
from app.ir.ir_graph import IRGraph
from app.ir.ir_node import IRNode
from app.codegen.layer_builders import build_layer, clean_variable_name

from app.codegen.generators.pytorch.generator import PyTorchCompiler

class PyTorchGenerator(BaseCompiler):
    """
    Code generator compiling framework-agnostic IRGraph or raw JSON dictionaries
    into fully runnable PyTorch scripts including both the Model class and Trainer.
    """
    
    def compile(self, ir_graph: IRGraph) -> str:
        """
        Compiles the IRGraph into fully runnable PyTorch code.
        """
        return PyTorchCompiler().compile(ir_graph)


    def compile_from_dict(self, graph_data: Dict[str, Any], project_name: str = "Project") -> str:
        """
        Traverses a dictionary of nodes/edges directly, builds intermediate structures, and compiles.
        """
        # Graph Traversal / Sort
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        node_by_id = {str(n.get("id") or n.get("node_id")): n for n in nodes if n.get("id") or n.get("node_id")}
        
        # Build adjacency maps
        adj = {nid: [] for nid in node_by_id}
        in_degree = {nid: 0 for nid in node_by_id}
        node_inputs = {nid: [] for nid in node_by_id}
        node_outputs = {nid: [] for nid in node_by_id}
        
        for e in edges:
            from_id = str(e.get("from") or e.get("from_node_id") or e.get("source") or "")
            to_id = str(e.get("to") or e.get("to_node_id") or e.get("target") or "")
            
            if from_id in node_by_id and to_id in node_by_id:
                adj[from_id].append(to_id)
                in_degree[to_id] += 1
                node_inputs[to_id].append(from_id)
                node_outputs[from_id].append(to_id)
                
        # Queue source nodes
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        queue.sort(key=lambda nid: 0 if str(node_by_id[nid].get("type") or node_by_id[nid].get("op_type", "")).lower() == "input" else 1)
        
        ordered_ids = []
        while queue:
            u = queue.pop(0)
            ordered_ids.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        if len(ordered_ids) != len(node_by_id):
            raise ValueError("Invalid Graph: Contains cyclic dependencies.")
            
        ordered_nodes = [node_by_id[nid] for nid in ordered_ids]
        
        # Construct IRGraph
        ir_graph = IRGraph(
            project_id="dict_project",
            project_name=project_name,
            framework="PyTorch"
        )
        
        for n in ordered_nodes:
            nid = str(n.get("id") or n.get("node_id"))
            ir_node = IRNode(
                id=nid,
                op_type=n.get("type") or n.get("op_type") or "Identity",
                label=n.get("label") or n.get("name") or nid,
                params=n.get("config") or n.get("params") or {},
                input_shape=n.get("input_shape") or n.get("inputShape"),
                output_shape=n.get("output_shape") or n.get("outputShape"),
                inputs=node_inputs[nid],
                outputs=node_outputs[nid]
            )
            ir_graph.add_node(ir_node)
            
        return self.compile(ir_graph)
