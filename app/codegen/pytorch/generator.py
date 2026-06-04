import os
import re
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict, Any

from app.codegen.base_compiler import BaseCompiler
from app.ir.ir_graph import IRGraph
from app.ir.ir_node import IRNode

class PyTorchCompiler(BaseCompiler):
    """PyTorch code generator compiling framework-agnostic IRGraph definitions into runnable PyTorch nn.Module scripts."""

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

    @staticmethod
    def get_activation_function(activation_str: Any) -> str | None:
        """Map config activation string to PyTorch/torch.nn.functional activations."""
        if not activation_str or not isinstance(activation_str, str):
            return None
        
        act = activation_str.lower().strip()
        if act == "relu":
            return "F.relu"
        elif act == "sigmoid":
            return "torch.sigmoid"
        elif act == "tanh":
            return "torch.tanh"
        elif act in ("softmax", "log_softmax"):
            return f"lambda t: F.{act}(t, dim=1)"
        return None

    def compile(self, ir_graph: IRGraph) -> str:
        """Compiles the IRGraph into fully runnable PyTorch code."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("model.py.jinja2")

        sorted_nodes = ir_graph.get_topologically_sorted_nodes()

        layers_init = []
        forward_steps = []
        
        # Track input shape details for dummy validation block
        input_shape_comment = "unknown"
        input_shape_test = "[1, 3, 224, 224]"
        test_input_dims = "1, 3, 224, 224"

        # Unique name tracker to avoid duplicate variable names
        name_counts: Dict[str, int] = {}
        # Keep track of generated variable names mapped by Node ID for forward pass connections
        node_vars: Dict[str, str] = {}

        # 1. Establish unique variable names for all nodes
        for node in sorted_nodes:
            op_type = node.op_type.lower()
            base_var = self.clean_variable_name(node.label or node.op_type)
            if base_var in name_counts:
                name_counts[base_var] += 1
                var_name = f"{base_var}_{name_counts[base_var]}"
            else:
                name_counts[base_var] = 1
                var_name = base_var
            
            node_vars[node.id] = var_name

        # 2. Iterate and map layers
        for node in sorted_nodes:
            op_type = node.op_type.lower()
            var_name = node_vars[node.id]

            if op_type == "input":
                shape = node.output_shape
                if shape:
                    input_shape_comment = str(shape)
                    test_dims = [dim if dim is not None else 1 for dim in shape]
                    input_shape_test = str(test_dims)
                    test_input_dims = ", ".join(map(str, test_dims))
                forward_steps.append({
                    "name": var_name,
                    "custom_forward": f"{var_name} = x",
                    "shape": str(node.output_shape),
                    "activation": None
                })
                continue

            params = node.params or {}
            
            # Resolve parent input variables
            # For multi-input operations (like Add or Multiply), we map them in the forward pass directly
            if not node.inputs:
                input_arg = "x"
            elif len(node.inputs) == 1:
                input_arg = node_vars[node.inputs[0]]
            else:
                parents = [node_vars[pid] for pid in node.inputs]
                input_arg = ", ".join(parents)

            # Map operations
            is_tensor_op = False
            custom_forward = None
            init_str = "nn.Identity()"
            activation = None

            if op_type == "conv2d":
                in_channels = node.input_shape[1] if node.input_shape else 3
                filters = int(params.get("filters", 32))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = params.get("padding", 0)

                padding_str = f"'{padding}'" if isinstance(padding, str) else str(padding)
                init_str = f"nn.Conv2d(in_channels={in_channels}, out_channels={filters}, kernel_size={kernel_size}, stride={stride}, padding={padding_str})"
                activation = self.get_activation_function(params.get("activation"))

            elif op_type in ("maxpool2d", "avgpool", "avgpool2d"):
                pool_size = params.get("pool_size", 2)
                stride = params.get("stride", pool_size)
                padding = params.get("padding", 0)

                pool_class = "MaxPool2d" if "max" in op_type else "AvgPool2d"
                init_str = f"nn.{pool_class}(kernel_size={pool_size}, stride={stride}, padding={padding})"

            elif op_type in ("adaptivepool", "adaptiveavgpool", "adaptiveavgpool2d", "adaptivemaxpool2d"):
                output_size = params.get("output_size", 1)
                pool_class = "AdaptiveMaxPool2d" if "max" in op_type else "AdaptiveAvgPool2d"
                init_str = f"nn.{pool_class}(output_size={output_size})"

            elif op_type in ("convtranspose", "convtranspose2d"):
                in_channels = node.input_shape[1] if node.input_shape else 3
                filters = int(params.get("filters", 32))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = params.get("padding", 0)
                output_padding = params.get("output_padding", 0)

                init_str = f"nn.ConvTranspose2d(in_channels={in_channels}, out_channels={filters}, kernel_size={kernel_size}, stride={stride}, padding={padding}, output_padding={output_padding})"

            elif op_type == "flatten":
                init_str = "nn.Flatten(start_dim=1)"

            elif op_type in ("dense", "linear"):
                in_features = node.input_shape[-1] if node.input_shape else 10
                units = int(params.get("units") or params.get("out_features", 10))

                init_str = f"nn.Linear(in_features={in_features}, out_features={units})"
                activation = self.get_activation_function(params.get("activation"))

            elif op_type in ("batchnorm", "batchnorm2d"):
                in_features = node.input_shape[1] if node.input_shape else 3
                init_str = f"nn.BatchNorm2d(num_features={in_features})"

            elif op_type in ("lstm", "gru", "rnn"):
                in_features = node.input_shape[2] if node.input_shape else 64
                hidden_size = int(params.get("hidden_size") or params.get("units", 64))
                
                rnn_class = "LSTM" if op_type == "lstm" else ("GRU" if op_type == "gru" else "RNN")
                init_str = f"nn.{rnn_class}(input_size={in_features}, hidden_size={hidden_size}, batch_first=True)"
                
                return_sequences = bool(params.get("return_sequences", False))
                # PyTorch RNNs return (output, states). We discard states for sequence mappings.
                if return_sequences:
                    custom_forward = f"{var_name}, _ = self.{var_name}({input_arg})"
                else:
                    custom_forward = f"_, (hn, _) = self.{var_name}({input_arg})\n        {var_name} = hn[-1]" if op_type == "lstm" else f"_, hn = self.{var_name}({input_arg})\n        {var_name} = hn[-1]"

            elif op_type == "bidirectional":
                in_features = node.input_shape[2] if node.input_shape else 64
                hidden_size = int(params.get("hidden_size") or params.get("units", 64))
                
                init_str = f"nn.LSTM(input_size={in_features}, hidden_size={hidden_size}, batch_first=True, bidirectional=True)"
                return_sequences = bool(params.get("return_sequences", False))
                if return_sequences:
                    custom_forward = f"{var_name}, _ = self.{var_name}({input_arg})"
                else:
                    custom_forward = f"_, (hn, _) = self.{var_name}({input_arg})\n        {var_name} = torch.cat((hn[-2], hn[-1]), dim=-1)"

            elif op_type == "embedding":
                vocab_size = int(params.get("input_dim") or params.get("vocab_size", 1000))
                embed_dim = int(params.get("embedding_dim") or params.get("output_dim", 128))
                init_str = f"nn.Embedding(num_embeddings={vocab_size}, embedding_dim={embed_dim})"

            elif op_type == "layernorm":
                normalized_shape = node.input_shape[1:] if node.input_shape else [64]
                init_str = f"nn.LayerNorm(normalized_shape={normalized_shape})"

            elif op_type in ("multiheadattention", "mha"):
                # Resolve embed_dim from config or query input shape
                embed_dim = params.get("embed_dim") or params.get("key_dim") or params.get("embedding_dim")
                if embed_dim is not None:
                    embed_dim = int(embed_dim)
                else:
                    q_shape = node.input_shape[0] if node.input_shape and isinstance(node.input_shape[0], list) else node.input_shape
                    embed_dim = q_shape[2] if q_shape and len(q_shape) > 2 else 128
                num_heads = int(params.get("num_heads", 8))
                init_str = f"nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={num_heads}, batch_first=True)"
                # MultiheadAttention requires Query, Key, Value inputs and returns output, weights.
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) == 3:
                    custom_forward = f"{var_name}, _ = self.{var_name}({parents[0]}, {parents[1]}, {parents[2]})"
                elif len(parents) == 2:
                    custom_forward = f"{var_name}, _ = self.{var_name}({parents[0]}, {parents[1]}, {parents[1]})"
                else:
                    custom_forward = f"{var_name}, _ = self.{var_name}({input_arg}, {input_arg}, {input_arg})"

            # Tensor Operations (No init definitions, custom forward steps)
            elif op_type == "add":
                is_tensor_op = True
                parents_vars = [node_vars[pid] for pid in node.inputs]
                sum_arg = " + ".join(parents_vars)
                custom_forward = f"{var_name} = {sum_arg}"

            elif op_type == "subtract":
                is_tensor_op = True
                parents_vars = [node_vars[pid] for pid in node.inputs]
                if len(parents_vars) >= 2:
                    sub_arg = " - ".join(parents_vars)
                    custom_forward = f"{var_name} = {sub_arg}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "maximum":
                is_tensor_op = True
                parents_vars = [node_vars[pid] for pid in node.inputs]
                if len(parents_vars) >= 2:
                    cur = parents_vars[0]
                    for other in parents_vars[1:]:
                        cur = f"torch.maximum({cur}, {other})"
                    custom_forward = f"{var_name} = {cur}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "minimum":
                is_tensor_op = True
                parents_vars = [node_vars[pid] for pid in node.inputs]
                if len(parents_vars) >= 2:
                    cur = parents_vars[0]
                    for other in parents_vars[1:]:
                        cur = f"torch.minimum({cur}, {other})"
                    custom_forward = f"{var_name} = {cur}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "multiply":
                is_tensor_op = True
                parents_vars = [node_vars[pid] for pid in node.inputs]
                mul_arg = " * ".join(parents_vars)
                custom_forward = f"{var_name} = {mul_arg}"

            elif op_type == "concatenate":
                is_tensor_op = True
                axis = int(params.get("axis", 1))
                parents_vars = [node_vars[pid] for pid in node.inputs]
                custom_forward = f"{var_name} = torch.cat(({', '.join(parents_vars)}), dim={axis})"

            elif op_type == "reshape":
                is_tensor_op = True
                target_shape = params.get("shape") or params.get("target_shape", [-1])
                # Format shape for PyTorch (e.g. replacing None/null in target_shape with -1)
                cleaned_shape = [dim if dim is not None else -1 for dim in target_shape]
                custom_forward = f"{var_name} = torch.reshape({input_arg}, {cleaned_shape})"

            elif op_type in ("permute", "transpose"):
                is_tensor_op = True
                axes = params.get("axes") or params.get("dims", [0, 2, 1, 3])
                custom_forward = f"{var_name} = {input_arg}.permute({axes})"
            else:
                # Default Identity fallback
                init_str = "nn.Identity()"

            # Append layer to dynamic init list
            if not is_tensor_op:
                layers_init.append({"name": var_name, "pytorch_init": init_str})

            # Append forward step
            if not custom_forward:
                custom_forward = f"{var_name} = self.{var_name}({input_arg})"

            forward_steps.append({
                "name": var_name,
                "custom_forward": custom_forward,
                "shape": str(node.output_shape),
                "activation": activation
            })

        # Render PyTorch Jinja2 Template
        # We need to adapt model.py.jinja2 to accept and render "custom_forward" if present!
        # Let's view the template contents first to verify if it has custom_forward or replace it!
        # Wait, the template has:
        # {% for step in forward_steps %}
        # x = self.{{ step.name }}(x)
        # {% endfor %}
        # Let's modify the template to render:
        # {% for step in forward_steps %}
        # {{ step.custom_forward }}  # shape: {{ step.shape }}
        # {% if step.activation %}
        # {{ step.name }} = {{ step.activation }}({{ step.name }})
        # {% endif %}
        # {% endfor %}
        # This is incredibly powerful!
        
        # Let's write the modified model.py.jinja2 first to support these custom forward steps!
        # The template path is d:\Coding\new_project\app\codegen\pytorch\templates\model.py.jinja2.
        
        return layers_init, forward_steps, input_shape_comment, input_shape_test, test_input_dims, template

    def compile(self, ir_graph: IRGraph) -> str:
        layers_init, forward_steps, input_shape_comment, input_shape_test, test_input_dims, template = self.compile_meta(ir_graph)
        
        rendered_code = template.render(
            project_name=ir_graph.project_name,
            framework=ir_graph.framework,
            layers_init=layers_init,
            forward_steps=forward_steps,
            input_shape_comment=input_shape_comment,
            input_shape_test=input_shape_test,
            test_input_dims=test_input_dims
        )
        return rendered_code

    def compile_meta(self, ir_graph: IRGraph) -> Any:
        return self.compile_layers(ir_graph)

    def compile_layers(self, ir_graph: IRGraph) -> Any:
        # This keeps it clean
        return self.compile_extracted(ir_graph)

    def compile_extracted(self, ir_graph: IRGraph) -> Any:
        # We split the extraction logic cleanly
        sorted_nodes = ir_graph.get_topologically_sorted_nodes()

        layers_init = []
        forward_steps = []
        
        input_shape_comment = "unknown"
        input_shape_test = "[1, 3, 224, 224]"
        test_input_dims = "1, 3, 224, 224"

        name_counts: Dict[str, int] = {}
        node_vars: Dict[str, str] = {}

        for node in sorted_nodes:
            op_type = node.op_type.lower()
            base_var = self.clean_variable_name(node.label or node.op_type)
            if base_var in name_counts:
                name_counts[base_var] += 1
                var_name = f"{base_var}_{name_counts[base_var]}"
            else:
                name_counts[base_var] = 1
                var_name = base_var
            node_vars[node.id] = var_name

        for node in sorted_nodes:
            op_type = node.op_type.lower()
            var_name = node_vars[node.id]

            if op_type == "input":
                shape = node.output_shape
                if shape:
                    input_shape_comment = str(shape)
                    test_dims = [dim if dim is not None else 1 for dim in shape]
                    input_shape_test = str(test_dims)
                    test_input_dims = ", ".join(map(str, test_dims))
                forward_steps.append({
                    "name": var_name,
                    "custom_forward": f"{var_name} = x",
                    "shape": str(node.output_shape),
                    "activation": None
                })
                continue

            params = node.params or {}
            
            if not node.inputs:
                input_arg = "x"
            elif len(node.inputs) == 1:
                input_arg = node_vars[node.inputs[0]]
            else:
                parents = [node_vars[pid] for pid in node.inputs]
                input_arg = ", ".join(parents)

            is_tensor_op = False
            custom_forward = None
            init_str = "nn.Identity()"
            activation = None

            if op_type == "conv2d":
                in_channels = node.input_shape[1] if node.input_shape else 3
                filters = int(params.get("filters", 32))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = params.get("padding", 0)
                padding_str = f"'{padding}'" if isinstance(padding, str) else str(padding)
                init_str = f"nn.Conv2d(in_channels={in_channels}, out_channels={filters}, kernel_size={kernel_size}, stride={stride}, padding={padding_str})"
                activation = self.get_activation_function(params.get("activation"))

            elif op_type in ("maxpool2d", "avgpool", "avgpool2d"):
                pool_size = params.get("pool_size", 2)
                stride = params.get("stride", pool_size)
                padding = params.get("padding", 0)
                pool_class = "MaxPool2d" if "max" in op_type else "AvgPool2d"
                init_str = f"nn.{pool_class}(kernel_size={pool_size}, stride={stride}, padding={padding})"

            elif op_type in ("adaptivepool", "adaptiveavgpool", "adaptiveavgpool2d", "adaptivemaxpool2d"):
                output_size = params.get("output_size", 1)
                pool_class = "AdaptiveMaxPool2d" if "max" in op_type else "AdaptiveAvgPool2d"
                init_str = f"nn.{pool_class}(output_size={output_size})"

            elif op_type in ("convtranspose", "convtranspose2d"):
                in_channels = node.input_shape[1] if node.input_shape else 3
                filters = int(params.get("filters", 32))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = params.get("padding", 0)
                output_padding = params.get("output_padding", 0)
                init_str = f"nn.ConvTranspose2d(in_channels={in_channels}, out_channels={filters}, kernel_size={kernel_size}, stride={stride}, padding={padding}, output_padding={output_padding})"

            elif op_type == "flatten":
                init_str = "nn.Flatten(start_dim=1)"

            elif op_type in ("dense", "linear"):
                in_features = node.input_shape[-1] if node.input_shape else 10
                units = int(params.get("units") or params.get("out_features", 10))
                init_str = f"nn.Linear(in_features={in_features}, out_features={units})"
                activation = self.get_activation_function(params.get("activation"))

            elif op_type in ("batchnorm", "batchnorm2d"):
                in_features = node.input_shape[1] if node.input_shape else 3
                init_str = f"nn.BatchNorm2d(num_features={in_features})"

            elif op_type in ("lstm", "gru", "rnn"):
                in_features = node.input_shape[2] if node.input_shape else 64
                hidden_size = int(params.get("hidden_size") or params.get("units", 64))
                rnn_class = "LSTM" if op_type == "lstm" else ("GRU" if op_type == "gru" else "RNN")
                init_str = f"nn.{rnn_class}(input_size={in_features}, hidden_size={hidden_size}, batch_first=True)"
                return_sequences = bool(params.get("return_sequences", False))
                if return_sequences:
                    custom_forward = f"{var_name}, _ = self.{var_name}({input_arg})"
                else:
                    custom_forward = f"_, (hn, _) = self.{var_name}({input_arg})\n        {var_name} = hn[-1]" if op_type == "lstm" else f"_, hn = self.{var_name}({input_arg})\n        {var_name} = hn[-1]"

            elif op_type == "bidirectional":
                in_features = node.input_shape[2] if node.input_shape else 64
                hidden_size = int(params.get("hidden_size") or params.get("units", 64))
                init_str = f"nn.LSTM(input_size={in_features}, hidden_size={hidden_size}, batch_first=True, bidirectional=True)"
                return_sequences = bool(params.get("return_sequences", False))
                if return_sequences:
                    custom_forward = f"{var_name}, _ = self.{var_name}({input_arg})"
                else:
                    custom_forward = f"_, (hn, _) = self.{var_name}({input_arg})\n        {var_name} = torch.cat((hn[-2], hn[-1]), dim=-1)"

            elif op_type == "embedding":
                vocab_size = int(params.get("input_dim") or params.get("vocab_size", 1000))
                embed_dim = int(params.get("embedding_dim") or params.get("output_dim", 128))
                init_str = f"nn.Embedding(num_embeddings={vocab_size}, embedding_dim={embed_dim})"

            elif op_type == "layernorm":
                normalized_shape = node.input_shape[1:] if node.input_shape else [64]
                init_str = f"nn.LayerNorm(normalized_shape={normalized_shape})"

            elif op_type in ("multiheadattention", "mha"):
                # Resolve embed_dim from config or query input shape
                embed_dim = params.get("embed_dim") or params.get("key_dim") or params.get("embedding_dim")
                if embed_dim is not None:
                    embed_dim = int(embed_dim)
                else:
                    if node.input_shape:
                        if isinstance(node.input_shape[0], list):
                            mha_input_shape = node.input_shape[0]
                        else:
                            mha_input_shape = node.input_shape
                        embed_dim = mha_input_shape[2] if len(mha_input_shape) > 2 else 128
                    else:
                        embed_dim = 128
                num_heads = int(params.get("num_heads", 8))
                init_str = f"nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={num_heads}, batch_first=True)"
                # MultiheadAttention requires Query, Key, Value inputs and returns output, weights.
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) == 3:
                    custom_forward = f"{var_name}, _ = self.{var_name}({parents[0]}, {parents[1]}, {parents[2]})"
                elif len(parents) == 2:
                    custom_forward = f"{var_name}, _ = self.{var_name}({parents[0]}, {parents[1]}, {parents[1]})"
                else:
                    custom_forward = f"{var_name}, _ = self.{var_name}({input_arg}, {input_arg}, {input_arg})"

            elif op_type == "add":
                is_tensor_op = True
                parents_vars = [node_vars[pid] for pid in node.inputs]
                custom_forward = f"{var_name} = {' + '.join(parents_vars)}"

            elif op_type == "subtract":
                is_tensor_op = True
                parents_vars = [node_vars[pid] for pid in node.inputs]
                if len(parents_vars) >= 2:
                    sub_arg = " - ".join(parents_vars)
                    custom_forward = f"{var_name} = {sub_arg}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "maximum":
                is_tensor_op = True
                parents_vars = [node_vars[pid] for pid in node.inputs]
                if len(parents_vars) >= 2:
                    cur = parents_vars[0]
                    for other in parents_vars[1:]:
                        cur = f"torch.maximum({cur}, {other})"
                    custom_forward = f"{var_name} = {cur}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "minimum":
                is_tensor_op = True
                parents_vars = [node_vars[pid] for pid in node.inputs]
                if len(parents_vars) >= 2:
                    cur = parents_vars[0]
                    for other in parents_vars[1:]:
                        cur = f"torch.minimum({cur}, {other})"
                    custom_forward = f"{var_name} = {cur}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "multiply":
                is_tensor_op = True
                parents_vars = [node_vars[pid] for pid in node.inputs]
                custom_forward = f"{var_name} = {' * '.join(parents_vars)}"

            elif op_type == "concatenate":
                is_tensor_op = True
                axis = int(params.get("axis", 1))
                parents_vars = [node_vars[pid] for pid in node.inputs]
                custom_forward = f"{var_name} = torch.cat(({', '.join(parents_vars)}), dim={axis})"

            elif op_type == "reshape":
                is_tensor_op = True
                target_shape = params.get("shape") or params.get("target_shape", [-1])
                cleaned_shape = [dim if dim is not None else -1 for dim in target_shape]
                custom_forward = f"{var_name} = torch.reshape({input_arg}, {cleaned_shape})"

            elif op_type in ("permute", "transpose"):
                is_tensor_op = True
                axes = params.get("axes") or params.get("dims", [0, 2, 1, 3])
                custom_forward = f"{var_name} = {input_arg}.permute({axes})"
            else:
                init_str = "nn.Identity()"

            if not is_tensor_op:
                layers_init.append({"name": var_name, "pytorch_init": init_str})

            if not custom_forward:
                custom_forward = f"{var_name} = self.{var_name}({input_arg})"

            forward_steps.append({
                "name": var_name,
                "custom_forward": custom_forward,
                "shape": str(node.output_shape),
                "activation": activation
            })

        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("model.py.jinja2")

        return layers_init, forward_steps, input_shape_comment, input_shape_test, test_input_dims, template
