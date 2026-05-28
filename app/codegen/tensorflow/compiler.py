import os
import re
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict, Any

from app.codegen.base_compiler import BaseCompiler
from app.ir.ir_graph import IRGraph
from app.ir.ir_node import IRNode

class TensorFlowCompiler(BaseCompiler):
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
                input_arg_str = "[" + ", ".join(parents_list) + "]"

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

            elif op_type == "layernorm":
                # Standard layer normalization along last axis
                keras_def = "layers.LayerNormalization(axis=-1)"

            elif op_type in ("multiheadattention", "mha"):
                num_heads = int(params.get("num_heads", 8))
                embed_dim = node.input_shape[2] if node.input_shape else 128
                keras_def = f"layers.MultiHeadAttention(num_heads={num_heads}, key_dim={embed_dim})"
                # In Keras, MultiHeadAttention takes query, value, key. Default to self-attention:
                is_custom_call = True
                custom_call_str = f"{var_name} = {keras_def}({input_arg_str}, {input_arg_str}, {input_arg_str})"

            elif op_type == "add":
                keras_def = "layers.Add()"

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
