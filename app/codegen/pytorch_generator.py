import os
import re
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader

from app.codegen.base_compiler import BaseCompiler
from app.ir.ir_graph import IRGraph
from app.ir.ir_node import IRNode
from app.codegen.layer_builders import build_layer, clean_variable_name

class PyTorchGenerator(BaseCompiler):
    """
    Code generator compiling framework-agnostic IRGraph or raw JSON dictionaries
    into fully runnable PyTorch scripts including both the Model class and Trainer.
    """
    
    def compile(self, ir_graph: IRGraph) -> str:
        """
        Compiles the IRGraph into fully runnable PyTorch code.
        """
        sorted_nodes = ir_graph.get_topologically_sorted_nodes()
        
        layers_init = []
        forward_steps = []
        
        input_shape_comment = "unknown"
        input_shape_test = "[1, 3, 224, 224]"
        test_input_dims = "1, 3, 224, 224"
        
        name_counts: Dict[str, int] = {}
        node_vars: Dict[str, str] = {}
        
        # 1. Map node IDs to clean variable names
        for node in sorted_nodes:
            base_var = clean_variable_name(node.label or node.op_type)
            if base_var in name_counts:
                name_counts[base_var] += 1
                var_name = f"{base_var}_{name_counts[base_var]}"
            else:
                name_counts[base_var] = 1
                var_name = base_var
            node_vars[node.id] = var_name

        # 2. Extract shape details for testing inputs from root Input node
        for node in sorted_nodes:
            if node.op_type.lower() == "input":
                shape = node.output_shape
                if shape:
                    input_shape_comment = str(shape)
                    test_dims = [dim if dim is not None else 1 for dim in shape]
                    input_shape_test = str(test_dims)
                    test_input_dims = ", ".join(map(str, test_dims))

        # 3. Build each layer using builders
        for node in sorted_nodes:
            init_str, forward_str, is_tensor_op, activation = build_layer(
                op_type=node.op_type,
                label=node.id,
                params=node.params or {},
                input_shape=node.input_shape,
                inputs=node.inputs,
                node_vars=node_vars
            )
            
            if not is_tensor_op and init_str:
                layers_init.append({
                    "name": node_vars[node.id],
                    "pytorch_init": init_str
                })
                
            if node.op_type.lower() != "input":
                forward_steps.append({
                    "name": node_vars[node.id],
                    "custom_forward": forward_str,
                    "shape": str(node.output_shape or "unknown"),
                    "activation": activation
                })

        # Load templates
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        model_template = env.get_template("model.py.j2")
        trainer_template = env.get_template("trainer.py.j2")

        # Render Model
        model_code = model_template.render(
            project_name=ir_graph.project_name,
            framework=ir_graph.framework,
            layers_init=layers_init,
            forward_steps=forward_steps,
            input_shape_comment=input_shape_comment,
            input_shape_test=input_shape_test,
            test_input_dims=test_input_dims
        )

        # Render Trainer
        trainer_code = trainer_template.render(
            test_input_dims=test_input_dims,
            input_shape_test=input_shape_test,
            forward_steps=forward_steps
        )

        # Combine them beautifully
        full_code = f"{model_code}\n\n# ==========================================================\n# TRAINING INFRASTRUCTURE\n# ==========================================================\n{trainer_code}"
        return full_code

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
