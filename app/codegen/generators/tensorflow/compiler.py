import os
import re
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict, Any

from app.codegen.base_compiler import BaseCompiler
from app.codegen.generators.registry import BaseGenerator
from app.ir.ir_graph import IRGraph
from app.ir.ir_node import IRNode

class TensorFlowCompiler(BaseGenerator, BaseCompiler):
    def generate(self, ir_graph: IRGraph) -> str:
        """Generates TensorFlow/Keras code conforming to the BaseGenerator contract."""
        return self.compile(ir_graph)
    """TensorFlow Keras Functional API code generator compiling framework-agnostic IRGraph definitions into runnable models."""

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
        """Compiles the IRGraph into fully runnable TensorFlow Keras code."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("model.tf.jinja2")

        sorted_nodes = ir_graph.get_topologically_sorted_nodes()

        layers_steps = []
        
        # Track input shape details
        input_variable = "x_input"
        input_name = "input_layer"
        input_shape_sans_batch = "(3, 224, 224)"
        test_input_dims = "1, 3, 224, 224"
        output_variable = "x_input"

        # Unique name tracker to avoid duplicate variable names
        name_counts: Dict[str, int] = {}
        node_vars: Dict[str, str] = {}

        # 1. Establish unique variable names for all nodes
        for node in sorted_nodes:
            op_type = node.op_type.lower()
            base_var = self.clean_variable_name(node.label or node.op_type)
            if base_var in name_counts:
                name_counts[base_var] += 1
                var_name = f"x_{base_var}_{name_counts[base_var]}"
            else:
                name_counts[base_var] = 1
                var_name = f"x_{base_var}"
            
            node_vars[node.id] = var_name

        # 2. Iterate and map layers
        for node in sorted_nodes:
            op_type = node.op_type.lower()
            var_name = node_vars[node.id]
            output_variable = var_name

            if op_type == "input":
                input_variable = var_name
                input_name = self.clean_variable_name(node.label)
                shape = node.output_shape
                if shape:
                    shape_without_batch = shape[1:]
                    input_shape_sans_batch = str(tuple(shape_without_batch))
                    
                    test_dims = [dim if dim is not None else 1 for dim in shape]
                    test_input_dims = ", ".join(map(str, test_dims))
                continue

            params = node.params or {}
            
            # Resolve parent input variables
            if not node.inputs:
                input_arg_str = ""
            elif len(node.inputs) == 1:
                input_arg_str = node_vars[node.inputs[0]]
            else:
                parents_list = [node_vars[pid] for pid in node.inputs]
                if op_type in (
                    "multiheadattention", "mha", "attention", "decoder_block", "residual_add", "concatenate",
                    "add", "subtract", "maximum", "minimum", "multiply"
                ):
                    input_arg_str = "[" + ", ".join(parents_list) + "]"
                else:
                    input_arg_str = f"layers.Concatenate(axis=1)([{', '.join(parents_list)}])"

            # Map operations
            keras_def = "layers.Activation('linear')"
            activation_name = None
            is_custom_call = False
            custom_call_str = ""

            if op_type == "conv2d":
                filters = int(params.get("filters", 32))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = params.get("padding", "valid")
                padding_str = padding.lower().strip() if isinstance(padding, str) else "valid"
                if padding_str not in ("valid", "same"):
                    padding_str = "valid"
                
                keras_def = f"layers.Conv2D(filters={filters}, kernel_size={kernel_size}, strides={stride}, padding='{padding_str}', data_format='channels_first')"
                if params.get("activation"):
                    activation_name = params.get("activation").lower().strip()

            elif op_type in ("maxpool2d", "avgpool", "avgpool2d"):
                pool_size = params.get("pool_size", 2)
                stride = params.get("stride", pool_size)
                padding = params.get("padding", "valid")
                padding_str = padding.lower().strip() if isinstance(padding, str) else "valid"
                if padding_str not in ("valid", "same"):
                    padding_str = "valid"

                pool_class = "MaxPooling2D" if "max" in op_type else "AveragePooling2D"
                keras_def = f"layers.{pool_class}(pool_size={pool_size}, strides={stride}, padding='{padding_str}', data_format='channels_first')"

            elif op_type in ("adaptivepool", "adaptiveavgpool", "adaptiveavgpool2d", "adaptivemaxpool2d"):
                # In Keras, global pooling is highly standard for adaptive outputs of [1, 1]
                pool_class = "GlobalMaxPooling2D" if "max" in op_type else "GlobalAveragePooling2D"
                keras_def = f"layers.{pool_class}(data_format='channels_first')"

            elif op_type in ("convtranspose", "convtranspose2d"):
                filters = int(params.get("filters", 32))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = params.get("padding", "valid")
                padding_str = padding.lower().strip() if isinstance(padding, str) else "valid"
                if padding_str not in ("valid", "same"):
                    padding_str = "valid"

                keras_def = f"layers.Conv2DTranspose(filters={filters}, kernel_size={kernel_size}, strides={stride}, padding='{padding_str}', data_format='channels_first')"

            elif op_type == "flatten":
                keras_def = "layers.Flatten(data_format='channels_first')"

            elif op_type in ("dense", "linear"):
                units = int(params.get("units") or params.get("out_features", 10))
                keras_def = f"layers.Dense(units={units})"
                if params.get("activation"):
                    activation_name = params.get("activation").lower().strip()

            elif op_type in ("batchnorm", "batchnorm2d"):
                # Axis 1 represents channels in channels_first
                keras_def = "layers.BatchNormalization(axis=1)"

            elif op_type in ("lstm", "gru", "rnn"):
                units = int(params.get("hidden_size") or params.get("units", 64))
                return_sequences = bool(params.get("return_sequences", False))
                rnn_class = "LSTM" if op_type == "lstm" else ("GRU" if op_type == "gru" else "SimpleRNN")
                keras_def = f"layers.{rnn_class}(units={units}, return_sequences={return_sequences})"

            elif op_type == "bidirectional":
                units = int(params.get("hidden_size") or params.get("units", 64))
                return_sequences = bool(params.get("return_sequences", False))
                keras_def = f"layers.Bidirectional(layers.LSTM(units={units}, return_sequences={return_sequences}))"

            elif op_type == "embedding":
                vocab_size = int(params.get("input_dim") or params.get("vocab_size", 1000))
                embed_dim = int(params.get("embedding_dim") or params.get("output_dim", 128))
                keras_def = f"layers.Embedding(input_dim={vocab_size}, output_dim={embed_dim})"

            elif op_type == "positional_encoding":
                embed_dim = params.get("embed_dim") or params.get("embedding_dim") or (node.input_shape[2] if node.input_shape and len(node.input_shape) > 2 else 128)
                max_len = params.get("max_len", 1000)
                is_custom_call = True
                custom_call_str = f"pos_indices_{var_name} = tf.range(tf.shape({input_arg_str})[1])[tf.newaxis, :]\n    pos_emb_{var_name} = layers.Embedding(input_dim={max_len}, output_dim={embed_dim})(pos_indices_{var_name})\n    {var_name} = layers.Add()([{input_arg_str}, pos_emb_{var_name}])"

            elif op_type in ("layernorm", "layer_norm"):
                # Standard layer normalization along last axis
                keras_def = "layers.LayerNormalization(axis=-1)"

            elif op_type in ("multiheadattention", "mha", "attention"):
                num_heads = int(params.get("num_heads", 8))
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
                keras_def = f"layers.MultiHeadAttention(num_heads={num_heads}, key_dim={embed_dim})"
                # In Keras, MultiHeadAttention takes query, value, key.
                is_custom_call = True
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) == 3:
                    custom_call_str = f"{var_name} = {keras_def}({parents[0]}, {parents[2]}, {parents[1]})"
                elif len(parents) == 2:
                    custom_call_str = f"{var_name} = {keras_def}({parents[0]}, {parents[1]}, {parents[1]})"
                else:
                    custom_call_str = f"{var_name} = {keras_def}({input_arg_str}, {input_arg_str}, {input_arg_str})"

            elif op_type == "residual_add":
                keras_def = "layers.Add()"

            elif op_type in ("transformer_block", "encoder_block"):
                embed_dim = params.get("embed_dim") or params.get("embedding_dim") or (node.input_shape[2] if node.input_shape and len(node.input_shape) > 2 else 128)
                num_heads = int(params.get("num_heads", 8))
                is_custom_call = True
                custom_call_str = f"mha_{var_name} = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={embed_dim})({input_arg_str}, {input_arg_str}, {input_arg_str})\n    norm1_{var_name} = layers.LayerNormalization()(layers.Add()([{input_arg_str}, mha_{var_name}]))\n    ffn_{var_name} = layers.Dense({embed_dim} * 4, activation='relu')(norm1_{var_name})\n    ffn_out_{var_name} = layers.Dense({embed_dim})(ffn_{var_name})\n    {var_name} = layers.LayerNormalization()(layers.Add()([norm1_{var_name}, ffn_out_{var_name}]))"

            elif op_type == "decoder_block":
                embed_dim = params.get("embed_dim") or params.get("embedding_dim") or (node.input_shape[0][2] if node.input_shape and isinstance(node.input_shape[0], list) and len(node.input_shape[0]) > 2 else 128)
                num_heads = int(params.get("num_heads", 8))
                is_custom_call = True
                parents = [node_vars[pid] for pid in node.inputs]
                target = parents[0] if len(parents) >= 1 else input_arg_str
                memory = parents[1] if len(parents) >= 2 else target
                custom_call_str = f"self_attn_{var_name} = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={embed_dim})({target}, {target}, {target})\n    norm1_{var_name} = layers.LayerNormalization()(layers.Add()([{target}, self_attn_{var_name}]))\n    cross_attn_{var_name} = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={embed_dim})(norm1_{var_name}, {memory}, {memory})\n    norm2_{var_name} = layers.LayerNormalization()(layers.Add()([norm1_{var_name}, cross_attn_{var_name}]))\n    ffn_{var_name} = layers.Dense({embed_dim} * 4, activation='relu')(norm2_{var_name})\n    ffn_out_{var_name} = layers.Dense({embed_dim})(ffn_{var_name})\n    {var_name} = layers.LayerNormalization()(layers.Add()([norm2_{var_name}, ffn_out_{var_name}]))"

            elif op_type == "bilstm":
                units = int(params.get("hidden_size") or params.get("units", 64))
                return_sequences = bool(params.get("return_sequences", True))
                keras_def = f"layers.Bidirectional(layers.LSTM(units={units}, return_sequences={return_sequences}))"

            elif op_type == "gcn":
                out_features = int(params.get("out_features") or params.get("units") or params.get("hidden_size") or params.get("features", 64))
                is_custom_call = True
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) >= 2:
                    custom_call_str = f"gcn_linear_{var_name} = layers.Dense(units={out_features})({parents[0]})\n    {var_name} = tf.matmul({parents[1]}, gcn_linear_{var_name})"
                else:
                    custom_call_str = f"{var_name} = layers.Dense(units={out_features})({input_arg_str})"

            elif op_type == "graph_sage":
                out_features = int(params.get("out_features") or params.get("units") or params.get("hidden_size") or params.get("features", 64))
                is_custom_call = True
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) >= 2:
                    custom_call_str = f"neigh_{var_name} = tf.matmul({parents[1]}, {parents[0]})\n    gcn_self_{var_name} = layers.Dense(units={out_features})({parents[0]})\n    gcn_neigh_{var_name} = layers.Dense(units={out_features})(neigh_{var_name})\n    {var_name} = layers.Add()([gcn_self_{var_name}, gcn_neigh_{var_name}])"
                else:
                    custom_call_str = f"{var_name} = layers.Dense(units={out_features})({input_arg_str})"

            elif op_type == "gat":
                out_features = int(params.get("out_features") or params.get("units") or params.get("hidden_size") or params.get("features", 64))
                is_custom_call = True
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) >= 2:
                    custom_call_str = f"gat_linear_{var_name} = layers.Dense(units={out_features})({parents[0]})\n    {var_name} = tf.matmul({parents[1]}, gat_linear_{var_name})"
                else:
                    custom_call_str = f"{var_name} = layers.Dense(units={out_features})({input_arg_str})"
            elif op_type == "add":
                keras_def = "layers.Add()"

            elif op_type == "subtract":
                keras_def = "layers.Subtract()"

            elif op_type == "maximum":
                keras_def = "layers.Maximum()"

            elif op_type == "minimum":
                keras_def = "layers.Minimum()"

            elif op_type == "multiply":
                keras_def = "layers.Multiply()"

            elif op_type == "concatenate":
                axis = int(params.get("axis", 1))
                keras_def = f"layers.Concatenate(axis={axis})"

            elif op_type == "reshape":
                # Keras Reshape does NOT expect batch size dimension in its initialization parameters
                target_shape = params.get("shape") or params.get("target_shape", [-1])
                # Filter out None/null batch dimension if present at index 0
                if target_shape[0] is None or target_shape[0] == -1:
                    target_shape_sans_batch = target_shape[1:]
                else:
                    target_shape_sans_batch = target_shape
                
                cleaned_shape = [dim if dim is not None else -1 for dim in target_shape_sans_batch]
                keras_def = f"layers.Reshape({tuple(cleaned_shape)})"

            elif op_type in ("permute", "transpose"):
                # Keras transpose op via direct TF calls
                is_custom_call = True
                axes = params.get("axes") or params.get("dims", [0, 2, 1, 3])
                custom_call_str = f"{var_name} = tf.transpose({input_arg_str}, perm={axes})"
            elif op_type == "dropout":
                rate = params.get("rate")
                if rate is None:
                    rate = params.get("p", 0.5)
                keras_def = f"layers.Dropout(rate={rate})"
            else:
                keras_def = "layers.Activation('linear')"

            if not is_custom_call:
                custom_call_str = f"{var_name} = {keras_def}({input_arg_str})"

            layers_steps.append({
                "output_var": var_name,
                "keras_definition": custom_call_str, # Will be rendered directly
                "input_vars": input_arg_str,
                "shape": str(node.output_shape),
                "activation": activation_name
            })

        # Render complete template
        rendered_code = template.render(
            project_name=ir_graph.project_name,
            project_clean_name=self.clean_variable_name(ir_graph.project_name),
            framework="TensorFlow/Keras",
            input_variable=input_variable,
            input_name=input_name,
            input_shape_sans_batch=input_shape_sans_batch,
            layers_steps=layers_steps,
            output_variable=output_variable,
            test_input_dims=test_input_dims
        )
        
        return rendered_code
