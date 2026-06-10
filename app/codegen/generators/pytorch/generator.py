import os
import re
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict, Any

from app.codegen.base_compiler import BaseCompiler
from app.codegen.generators.registry import BaseGenerator
from app.ir.ir_graph import IRGraph
from app.ir.ir_node import IRNode


class PyTorchCompiler(BaseGenerator, BaseCompiler):
    """PyTorch code generator compiling framework-agnostic IRGraph definitions into runnable PyTorch nn.Module scripts."""

    def generate(self, ir_graph: IRGraph) -> str:
        """Generates PyTorch code conforming to the BaseGenerator contract."""
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
        """Compiles the IRGraph into fully runnable PyTorch nn.Module code."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("model.py.jinja2")

        sorted_nodes = ir_graph.get_topologically_sorted_nodes()

        layers_init: List[Dict] = []
        forward_steps: List[Dict] = []

        input_shape_comment = "unknown"
        input_shape_test = "[1, 3, 224, 224]"
        test_input_dims = "1, 3, 224, 224"

        # Unique name tracker to avoid duplicate variable names
        name_counts: Dict[str, int] = {}
        # Map node ID -> variable name for forward pass wiring
        node_vars: Dict[str, str] = {}

        # Pass 1: Establish unique variable names for all nodes
        for node in sorted_nodes:
            base_var = self.clean_variable_name(node.label or node.op_type)
            if base_var in name_counts:
                name_counts[base_var] += 1
                var_name = f"{base_var}_{name_counts[base_var]}"
            else:
                name_counts[base_var] = 1
                var_name = base_var
            node_vars[node.id] = var_name

        # Pass 2: Map each node to a layer init and forward step
        for node in sorted_nodes:
            op_type = node.op_type.lower()
            var_name = node_vars[node.id]

            # ── Input node ────────────────────────────────────────────────────
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

            # Resolve upstream variable name(s)
            if not node.inputs:
                input_arg = "x"
            elif len(node.inputs) == 1:
                input_arg = node_vars[node.inputs[0]]
            else:
                parents = [node_vars[pid] for pid in node.inputs]
                if op_type in ("multiheadattention", "mha", "attention", "decoder_block", "residual_add", "concatenate"):
                    input_arg = ", ".join(parents)
                else:
                    input_arg = f"torch.cat([{', '.join(parents)}], dim=1)"

            is_tensor_op = False
            custom_forward = None
            init_str = "nn.Identity()"
            activation = None

            # ── Conv2D ────────────────────────────────────────────────────────
            if op_type == "conv2d":
                in_channels = node.input_shape[1] if node.input_shape and len(node.input_shape) > 1 else 3
                filters = int(params.get("filters", 32))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = params.get("padding", 0)
                padding_str = f"'{padding}'" if isinstance(padding, str) else str(padding)
                init_str = (
                    f"nn.Conv2d(in_channels={in_channels}, out_channels={filters}, "
                    f"kernel_size={kernel_size}, stride={stride}, padding={padding_str})"
                )
                activation = self.get_activation_function(params.get("activation"))

            # ── Pooling ───────────────────────────────────────────────────────
            elif op_type in (
                "maxpool", "maxpool2d", "maxpooling2d", "max_pooling2d",
                "avgpool", "avgpool2d", "avgpooling2d", "avg_pooling2d"
            ):
                pool_size = params.get("pool_size", 2)
                stride = params.get("stride", pool_size)
                padding = params.get("padding", 0)
                pool_class = "MaxPool2d" if "max" in op_type else "AvgPool2d"
                init_str = f"nn.{pool_class}(kernel_size={pool_size}, stride={stride}, padding={padding})"

            # ── Adaptive Pooling ──────────────────────────────────────────────
            elif op_type in (
                "adaptivepool", "adaptiveavgpool", "adaptiveavgpool2d", "adaptivemaxpool2d"
            ):
                output_size = params.get("output_size", 1)
                pool_class = "AdaptiveMaxPool2d" if "max" in op_type else "AdaptiveAvgPool2d"
                init_str = f"nn.{pool_class}(output_size={output_size})"

            # ── ConvTranspose ─────────────────────────────────────────────────
            elif op_type in ("convtranspose", "convtranspose2d"):
                in_channels = node.input_shape[1] if node.input_shape and len(node.input_shape) > 1 else 3
                filters = int(params.get("filters", 32))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = params.get("padding", 0)
                output_padding = params.get("output_padding", 0)
                init_str = (
                    f"nn.ConvTranspose2d(in_channels={in_channels}, out_channels={filters}, "
                    f"kernel_size={kernel_size}, stride={stride}, padding={padding}, "
                    f"output_padding={output_padding})"
                )

            # ── Flatten ───────────────────────────────────────────────────────
            elif op_type == "flatten":
                init_str = "nn.Flatten(start_dim=1)"

            # ── Dense / Linear ────────────────────────────────────────────────
            elif op_type in ("dense", "linear"):
                in_features = node.input_shape[-1] if node.input_shape else 10
                if in_features is None:
                    in_features = 10
                units = int(params.get("units") or params.get("out_features", 10))
                init_str = f"nn.Linear(in_features={in_features}, out_features={units})"
                activation = self.get_activation_function(params.get("activation"))

            # ── BatchNorm ─────────────────────────────────────────────────────
            elif op_type in ("batchnorm", "batchnorm2d"):
                in_features = node.input_shape[1] if node.input_shape and len(node.input_shape) > 1 else 3
                init_str = f"nn.BatchNorm2d(num_features={in_features})"

            # ── Dropout ───────────────────────────────────────────────────────
            elif op_type == "dropout":
                p = params.get("p") or params.get("rate", 0.5)
                init_str = f"nn.Dropout(p={p})"

            # ── LSTM / GRU / RNN ──────────────────────────────────────────────
            elif op_type in ("lstm", "gru", "rnn"):
                in_features = node.input_shape[2] if node.input_shape and len(node.input_shape) > 2 else 64
                hidden_size = int(params.get("hidden_size") or params.get("units", 64))
                rnn_class = "LSTM" if op_type == "lstm" else ("GRU" if op_type == "gru" else "RNN")
                init_str = f"nn.{rnn_class}(input_size={in_features}, hidden_size={hidden_size}, batch_first=True)"
                return_sequences = bool(params.get("return_sequences", False))
                if return_sequences:
                    custom_forward = f"{var_name}, _ = self.{var_name}({input_arg})"
                else:
                    if op_type == "lstm":
                        custom_forward = (
                            f"_, (hn_{var_name}, _) = self.{var_name}({input_arg})\n"
                            f"        {var_name} = hn_{var_name}[-1]"
                        )
                    else:
                        custom_forward = (
                            f"_, hn_{var_name} = self.{var_name}({input_arg})\n"
                            f"        {var_name} = hn_{var_name}[-1]"
                        )

            # ── Bidirectional ─────────────────────────────────────────────────
            elif op_type == "bidirectional":
                in_features = node.input_shape[2] if node.input_shape and len(node.input_shape) > 2 else 64
                hidden_size = int(params.get("hidden_size") or params.get("units", 64))
                init_str = f"nn.LSTM(input_size={in_features}, hidden_size={hidden_size}, batch_first=True, bidirectional=True)"
                return_sequences = bool(params.get("return_sequences", False))
                if return_sequences:
                    custom_forward = f"{var_name}, _ = self.{var_name}({input_arg})"
                else:
                    custom_forward = (
                        f"_, (hn_{var_name}, _) = self.{var_name}({input_arg})\n"
                        f"        {var_name} = torch.cat((hn_{var_name}[-2], hn_{var_name}[-1]), dim=-1)"
                    )

            # ── Embedding ─────────────────────────────────────────────────────
            elif op_type == "embedding":
                vocab_size = int(params.get("input_dim") or params.get("vocab_size", 1000))
                embed_dim = int(params.get("embedding_dim") or params.get("output_dim", 128))
                init_str = f"nn.Embedding(num_embeddings={vocab_size}, embedding_dim={embed_dim})"

            # ── Positional Encoding ───────────────────────────────────────────
            elif op_type == "positional_encoding":
                embed_dim = params.get("embed_dim") or params.get("embedding_dim") or (node.input_shape[2] if node.input_shape and len(node.input_shape) > 2 else 128)
                max_len = params.get("max_len", 1000)
                init_str = f"nn.Parameter(torch.randn(1, {max_len}, {embed_dim}))"
                custom_forward = f"{var_name} = {input_arg} + self.{var_name}[:, :{input_arg}.size(1)]"

            # ── LayerNorm ─────────────────────────────────────────────────────
            elif op_type in ("layernorm", "layer_norm"):
                normalized_shape = node.input_shape[1:] if node.input_shape else [64]
                init_str = f"nn.LayerNorm(normalized_shape={normalized_shape})"

            # ── MultiheadAttention / Attention ────────────────────────────────
            elif op_type in ("multiheadattention", "mha", "attention"):
                embed_dim = params.get("embed_dim") or params.get("key_dim") or params.get("embedding_dim")
                if embed_dim is not None:
                    embed_dim = int(embed_dim)
                else:
                    if node.input_shape:
                        q_shape = (
                            node.input_shape[0]
                            if isinstance(node.input_shape[0], list)
                            else node.input_shape
                        )
                        embed_dim = q_shape[2] if q_shape and len(q_shape) > 2 else 128
                    else:
                        embed_dim = 128
                num_heads = int(params.get("num_heads", 8))
                init_str = f"nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={num_heads}, batch_first=True)"
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) == 3:
                    custom_forward = f"{var_name}, _ = self.{var_name}({parents[0]}, {parents[1]}, {parents[2]})"
                elif len(parents) == 2:
                    custom_forward = f"{var_name}, _ = self.{var_name}({parents[0]}, {parents[1]}, {parents[1]})"
                else:
                    custom_forward = f"{var_name}, _ = self.{var_name}({input_arg}, {input_arg}, {input_arg})"

            # ── Residual Add ──────────────────────────────────────────────────
            elif op_type == "residual_add":
                is_tensor_op = True
                parents = [node_vars[pid] for pid in node.inputs]
                custom_forward = f"{var_name} = {' + '.join(parents)}"

            # ── Transformer Block / Encoder Block ─────────────────────────────
            elif op_type in ("transformer_block", "encoder_block"):
                embed_dim = params.get("embed_dim") or params.get("embedding_dim") or (node.input_shape[2] if node.input_shape and len(node.input_shape) > 2 else 128)
                num_heads = int(params.get("num_heads", 8))
                init_str = f"nn.TransformerEncoderLayer(d_model={embed_dim}, nhead={num_heads}, batch_first=True)"

            # ── Decoder Block ──────────────────────────────────────────────────
            elif op_type == "decoder_block":
                embed_dim = params.get("embed_dim") or params.get("embedding_dim") or (node.input_shape[0][2] if node.input_shape and isinstance(node.input_shape[0], list) and len(node.input_shape[0]) > 2 else 128)
                num_heads = int(params.get("num_heads", 8))
                init_str = f"nn.TransformerDecoderLayer(d_model={embed_dim}, nhead={num_heads}, batch_first=True)"
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) >= 2:
                    custom_forward = f"{var_name} = self.{var_name}({parents[0]}, {parents[1]})"
                else:
                    custom_forward = f"{var_name} = self.{var_name}({input_arg}, {input_arg})"

            # ── Bidirectional LSTM (BiLSTM) ───────────────────────────────────
            elif op_type == "bilstm":
                in_features = node.input_shape[2] if node.input_shape and len(node.input_shape) > 2 else 64
                hidden_size = int(params.get("hidden_size") or params.get("units", 64))
                num_layers = int(params.get("num_layers", 1))
                init_str = f"nn.LSTM(input_size={in_features}, hidden_size={hidden_size}, num_layers={num_layers}, batch_first=True, bidirectional=True)"
                return_sequences = bool(params.get("return_sequences", True))
                if return_sequences:
                    custom_forward = f"{var_name}, _ = self.{var_name}({input_arg})"
                else:
                    custom_forward = (
                        f"_, (hn_{var_name}, _) = self.{var_name}({input_arg})\n"
                        f"        {var_name} = torch.cat((hn_{var_name}[-2], hn_{var_name}[-1]), dim=-1)"
                    )

            # ── Graph Neural Networks (GCN / SAGE / GAT) ─────────────────────
            elif op_type == "gcn":
                in_features = node.input_shape[0][-1] if isinstance(node.input_shape[0], list) else (node.input_shape[-1] if node.input_shape else 64)
                out_features = int(params.get("out_features") or params.get("units") or params.get("hidden_size") or params.get("features", 64))
                init_str = f"GCNConv(in_channels={in_features}, out_channels={out_features})"
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) >= 2:
                    custom_forward = f"{var_name} = self.{var_name}({parents[0]}, {parents[1]})"
                else:
                    custom_forward = f"{var_name} = self.{var_name}({input_arg}, {input_arg})"

            elif op_type == "graph_sage":
                in_features = node.input_shape[0][-1] if isinstance(node.input_shape[0], list) else (node.input_shape[-1] if node.input_shape else 64)
                out_features = int(params.get("out_features") or params.get("units") or params.get("hidden_size") or params.get("features", 64))
                init_str = f"SAGEConv(in_channels={in_features}, out_channels={out_features})"
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) >= 2:
                    custom_forward = f"{var_name} = self.{var_name}({parents[0]}, {parents[1]})"
                else:
                    custom_forward = f"{var_name} = self.{var_name}({input_arg}, {input_arg})"

            elif op_type == "gat":
                in_features = node.input_shape[0][-1] if isinstance(node.input_shape[0], list) else (node.input_shape[-1] if node.input_shape else 64)
                out_features = int(params.get("out_features") or params.get("units") or params.get("hidden_size") or params.get("features", 64))
                heads = int(params.get("num_heads", 1))
                init_str = f"GATConv(in_channels={in_features}, out_channels={out_features}, heads={heads})"
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) >= 2:
                    custom_forward = f"{var_name} = self.{var_name}({parents[0]}, {parents[1]})"
                else:
                    custom_forward = f"{var_name} = self.{var_name}({input_arg}, {input_arg})"

            # ── Tensor Ops ────────────────────────────────────────────────────
            elif op_type == "add":
                is_tensor_op = True
                pv = [node_vars[pid] for pid in node.inputs]
                custom_forward = f"{var_name} = {' + '.join(pv)}"

            elif op_type == "subtract":
                is_tensor_op = True
                pv = [node_vars[pid] for pid in node.inputs]
                custom_forward = f"{var_name} = {' - '.join(pv)}" if len(pv) >= 2 else f"{var_name} = {input_arg}"

            elif op_type == "maximum":
                is_tensor_op = True
                pv = [node_vars[pid] for pid in node.inputs]
                if len(pv) >= 2:
                    cur = pv[0]
                    for other in pv[1:]:
                        cur = f"torch.maximum({cur}, {other})"
                    custom_forward = f"{var_name} = {cur}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "minimum":
                is_tensor_op = True
                pv = [node_vars[pid] for pid in node.inputs]
                if len(pv) >= 2:
                    cur = pv[0]
                    for other in pv[1:]:
                        cur = f"torch.minimum({cur}, {other})"
                    custom_forward = f"{var_name} = {cur}"
                else:
                    custom_forward = f"{var_name} = {input_arg}"

            elif op_type == "multiply":
                is_tensor_op = True
                pv = [node_vars[pid] for pid in node.inputs]
                custom_forward = f"{var_name} = {' * '.join(pv)}"

            elif op_type == "concatenate":
                is_tensor_op = True
                axis = int(params.get("axis", 1))
                pv = [node_vars[pid] for pid in node.inputs]
                custom_forward = f"{var_name} = torch.cat(({', '.join(pv)}), dim={axis})"

            elif op_type == "reshape":
                is_tensor_op = True
                target_shape = params.get("shape") or params.get("target_shape", [-1])
                cleaned = [dim if dim is not None else -1 for dim in target_shape]
                custom_forward = f"{var_name} = torch.reshape({input_arg}, {cleaned})"

            elif op_type in ("permute", "transpose"):
                is_tensor_op = True
                axes = params.get("axes") or params.get("dims", [0, 2, 1, 3])
                custom_forward = f"{var_name} = {input_arg}.permute({axes})"

            # ── Default Identity fallback ──────────────────────────────────────
            else:
                init_str = "nn.Identity()"

            # Register layer init (for non-tensor ops)
            if not is_tensor_op:
                layers_init.append({"name": var_name, "pytorch_init": init_str})

            # Build forward step string
            if not custom_forward:
                custom_forward = f"{var_name} = self.{var_name}({input_arg})"

            forward_steps.append({
                "name": var_name,
                "custom_forward": custom_forward,
                "shape": str(node.output_shape),
                "activation": activation
            })

        # Render Jinja2 template
        rendered_code = template.render(
            project_name=ir_graph.project_name,
            framework=ir_graph.framework,
            layers_init=layers_init,
            forward_steps=forward_steps,
            input_shape_comment=input_shape_comment,
            input_shape_test=input_shape_test,
            test_input_dims=test_input_dims,
            has_gnn=any(n.op_type.lower() in ("gcn", "graph_sage", "gat") for n in sorted_nodes)
        )
        return rendered_code
