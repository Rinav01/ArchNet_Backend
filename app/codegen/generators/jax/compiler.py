import os
import re
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict, Any

from app.codegen.base_compiler import BaseCompiler
from app.codegen.generators.registry import BaseGenerator
from app.ir.ir_graph import IRGraph
from app.ir.ir_node import IRNode

class JAXCompiler(BaseGenerator, BaseCompiler):
    """JAX/Flax Linen code generator compiling framework-agnostic IRGraph definitions into runnable Flax nn.Module scripts."""

    def generate(self, ir_graph: IRGraph) -> str:
        """Generates JAX/Flax code conforming to the BaseGenerator contract."""
        return self.compile(ir_graph)


    @staticmethod
    def clean_variable_name(label: str) -> str:
        """Sanitize a label to make it a valid Python variable name."""
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', label.strip())
        cleaned = re.sub(r'_+', '_', cleaned).lower()
        if not cleaned:
            return "layer"
        if cleaned[0].isdigit():
            cleaned = "layer_" + cleaned
        return cleaned

    def compile(self, ir_graph: IRGraph) -> str:
        """Compiles the IRGraph into fully runnable JAX/Flax code."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("model.jax.jinja2")

        sorted_nodes = ir_graph.get_topologically_sorted_nodes()

        forward_steps = []
        
        # Track input shape details for dummy validation block
        # Map channels_first input shape to JAX's default channels_last format
        test_input_dims = None
        input_node = next((n for n in sorted_nodes if n.op_type.lower() == "input"), None)
        if input_node and input_node.output_shape:
            raw_shape = input_node.output_shape
            dims = [dim if dim is not None else 1 for dim in raw_shape]
            if len(dims) == 4:
                # Swaps channels_first [Batch, Channels, Height, Width] -> channels_last [Batch, Height, Width, Channels]
                if dims[1] < dims[3] or dims[1] in (1, 3, 4):
                    jax_dims = [dims[0], dims[2], dims[3], dims[1]]
                else:
                    jax_dims = dims
            else:
                jax_dims = dims
            test_input_dims = ", ".join(map(str, jax_dims))

        # Unique name tracker to avoid duplicate variable names
        name_counts: Dict[str, int] = {}
        # Keep track of generated variable names mapped by Node ID for forward pass connections
        node_vars: Dict[str, str] = {}

        # 1. Establish unique variable names for all nodes
        for node in sorted_nodes:
            base_var = self.clean_variable_name(node.label or node.op_type)
            if base_var in name_counts:
                name_counts[base_var] += 1
                var_name = f"{base_var}_{name_counts[base_var]}"
            else:
                name_counts[base_var] = 1
                var_name = base_var
            
            node_vars[node.id] = var_name

        # Track outgoing counts to locate leaf nodes
        outgoing_count: Dict[str, int] = {nid: 0 for nid in node_vars}
        for node in sorted_nodes:
            for pid in node.inputs:
                if pid in outgoing_count:
                    outgoing_count[pid] += 1

        # 2. Iterate and map layers
        for node in sorted_nodes:
            op_type = node.op_type.lower()
            var_name = node_vars[node.id]
            params = node.params or {}

            if op_type == "input":
                forward_steps.append({
                    "name": var_name,
                    "custom_forward": f"{var_name} = x",
                    "shape": str(node.output_shape),
                    "activation": None
                })
                continue

            # Resolve parent input variables
            if not node.inputs:
                input_arg = "x"
            elif len(node.inputs) == 1:
                input_arg = node_vars[node.inputs[0]]
            else:
                parents = [node_vars[pid] for pid in node.inputs]
                # Default fallback for multiple inputs in general layers
                input_arg = f"jnp.concatenate([{', '.join(parents)}], axis=-1)"

            custom_forward = None
            activation = None

            if op_type == "conv2d":
                filters = int(params.get("filters", 64))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = str(params.get("padding", "SAME")).upper()
                if padding not in ("SAME", "VALID"):
                    padding = "SAME"
                
                act_str = params.get("activation")
                act_suffix = ""
                if act_str and str(act_str).lower() != "none":
                    act_suffix = f", activation=nn.{str(act_str).lower()}"

                custom_forward = f"{var_name} = nn.Conv(features={filters}, kernel_size=({kernel_size}, {kernel_size}), strides=({stride}, {stride}), padding='{padding}'{act_suffix})({input_arg})"

            elif op_type in ("maxpool2d", "avgpool", "avgpool2d"):
                pool_size = params.get("pool_size", 2)
                stride = params.get("stride", pool_size)
                
                pool_fn = "max_pool" if "max" in op_type else "avg_pool"
                custom_forward = f"{var_name} = nn.{pool_fn}({input_arg}, window_shape=({pool_size}, {pool_size}), strides=({stride}, {stride}))"

            elif op_type in ("adaptivepool", "adaptiveavgpool", "adaptiveavgpool2d", "adaptivemaxpool2d"):
                # Flax doesn't have direct Adaptive Pooling, we fallback to a standard global average pool along spatial axes
                # Assuming input shape is [Batch, Height, Width, Channels]
                custom_forward = f"{var_name} = jnp.mean({input_arg}, axis=(1, 2), keepdims=True)"

            elif op_type in ("convtranspose", "convtranspose2d"):
                filters = int(params.get("filters", 64))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = str(params.get("padding", "SAME")).upper()
                if padding not in ("SAME", "VALID"):
                    padding = "SAME"
                
                custom_forward = f"{var_name} = nn.ConvTranspose(features={filters}, kernel_size=({kernel_size}, {kernel_size}), strides=({stride}, {stride}), padding='{padding}')({input_arg})"

            elif op_type == "flatten":
                custom_forward = f"{var_name} = {input_arg}.reshape(({input_arg}.shape[0], -1))"

            elif op_type in ("dense", "linear"):
                units = int(params.get("units") or params.get("out_features", 10))
                act_str = params.get("activation")
                act_suffix = ""
                if act_str and str(act_str).lower() != "none":
                    act_suffix = f", activation=nn.{str(act_str).lower()}"
                
                custom_forward = f"{var_name} = nn.Dense(features={units}{act_suffix})({input_arg})"

            elif op_type in ("batchnorm", "batchnorm2d"):
                custom_forward = f"{var_name} = nn.BatchNorm()({input_arg})"

            elif op_type in ("lstm", "gru", "rnn", "bidirectional"):
                # Flax RNNs require a stateful API or Cell wrappers.
                # For code generation, we emit a functional mapping of nn.OptimizedLSTM/nn.GRUCell
                units = int(params.get("hidden_size") or params.get("units", 64))
                rnn_class = "OptimizedLSTM" if op_type == "lstm" or op_type == "bidirectional" else "GRUCell"
                
                # Dynamic shape tracking helper
                custom_forward = f"# Stateful recurrent layer mapping\n        {var_name}_layer = nn.{rnn_class}(features={units})\n        {var_name}, _ = {var_name}_layer({input_arg})"

            elif op_type == "dropout":
                rate = params.get("rate")
                if rate is None:
                    rate = params.get("p", 0.5)
                custom_forward = f"{var_name} = nn.Dropout(rate={rate}, deterministic=True)({input_arg})"

            elif op_type in ("multiheadattention", "mha"):
                num_heads = int(params.get("num_heads", 8))
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) == 3:
                    custom_forward = f"{var_name} = nn.MultiHeadDotProductAttention(num_heads={num_heads})({parents[0]}, jnp.concatenate([{parents[1]}, {parents[2]}], axis=-1))"
                elif len(parents) == 2:
                    custom_forward = f"{var_name} = nn.MultiHeadDotProductAttention(num_heads={num_heads})({parents[0]}, {parents[1]})"
                else:
                    custom_forward = f"{var_name} = nn.MultiHeadDotProductAttention(num_heads={num_heads})({input_arg})"

            elif op_type == "add":
                parents_vars = [node_vars[pid] for pid in node.inputs]
                sum_arg = " + ".join(parents_vars)
                custom_forward = f"{var_name} = {sum_arg}"

            elif op_type == "subtract":
                parents_vars = [node_vars[pid] for pid in node.inputs]
                if len(parents_vars) >= 2:
                    sub_arg = " - ".join(parents_vars)
                    custom_forward = f"{var_name} = {sub_arg}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "maximum":
                parents_vars = [node_vars[pid] for pid in node.inputs]
                if len(parents_vars) >= 2:
                    cur = parents_vars[0]
                    for other in parents_vars[1:]:
                        cur = f"jnp.maximum({cur}, {other})"
                    custom_forward = f"{var_name} = {cur}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "minimum":
                parents_vars = [node_vars[pid] for pid in node.inputs]
                if len(parents_vars) >= 2:
                    cur = parents_vars[0]
                    for other in parents_vars[1:]:
                        cur = f"jnp.minimum({cur}, {other})"
                    custom_forward = f"{var_name} = {cur}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "multiply":
                parents_vars = [node_vars[pid] for pid in node.inputs]
                mul_arg = " * ".join(parents_vars)
                custom_forward = f"{var_name} = {mul_arg}"

            elif op_type == "concatenate":
                axis = int(params.get("axis", -1))
                parents_vars = [node_vars[pid] for pid in node.inputs]
                custom_forward = f"{var_name} = jnp.concatenate([{', '.join(parents_vars)}], axis={axis})"

            elif op_type == "reshape":
                target_shape = params.get("shape") or params.get("target_shape", [-1])
                cleaned_shape = [dim if dim is not None else -1 for dim in target_shape]
                custom_forward = f"{var_name} = jnp.reshape({input_arg}, {tuple(cleaned_shape)})"

            elif op_type in ("permute", "transpose"):
                axes = params.get("axes") or params.get("dims", [0, 2, 1, 3])
                custom_forward = f"{var_name} = jnp.transpose({input_arg}, axes={axes})"

            else:
                # Identity fallback
                custom_forward = f"{var_name} = {input_arg}  # Identity fallback"

            forward_steps.append({
                "name": var_name,
                "custom_forward": custom_forward,
                "shape": str(node.output_shape),
                "activation": None
            })

        # Calculate final returns
        final_leaves = [node_vars[nid] for nid, count in outgoing_count.items() if count == 0]
        if not final_leaves:
            output_var = "x"
        elif len(final_leaves) == 1:
            output_var = final_leaves[0]
        else:
            output_var = ", ".join(final_leaves)

        # Render complete template
        rendered_code = template.render(
            project_name=ir_graph.project_name,
            framework="JAX/Flax",
            forward_steps=forward_steps,
            output_var=output_var,
            test_input_dims=test_input_dims
        )
        
        return rendered_code
