import os
import re
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict, Any

from app.codegen.base_compiler import BaseCompiler
from app.codegen.generators.registry import BaseGenerator
from app.ir.ir_graph import IRGraph
from app.ir.ir_node import IRNode

class ONNXCompiler(BaseGenerator, BaseCompiler):
    """ONNX code generator compiling framework-agnostic IRGraph definitions into runnable ONNX helper scripts."""

    def generate(self, ir_graph: IRGraph) -> str:
        """Generates ONNX helper python code conforming to the BaseGenerator contract."""
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
        """Compiles the IRGraph into fully runnable ONNX graph construction code."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("model.onnx.jinja2")

        sorted_nodes = ir_graph.get_topologically_sorted_nodes()

        initializers = []
        helper_nodes = []
        
        # Track input shape details
        input_dims = [1, 3, 224, 224]
        input_node = next((n for n in sorted_nodes if n.op_type.lower() == "input"), None)
        if input_node and input_node.output_shape:
            raw_shape = input_node.output_shape
            input_dims = [dim if dim is not None else 1 for dim in raw_shape]

        # Unique name tracker to avoid duplicate variable names
        name_counts: Dict[str, int] = {}
        # Keep track of generated variable names mapped by Node ID
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
                node_vars[node.id] = "x"  # The base input tensor is named "x"
                continue

            # Resolve parent input variables
            if not node.inputs:
                input_arg = "x"
            elif len(node.inputs) == 1:
                input_arg = node_vars[node.inputs[0]]
            else:
                parents = [node_vars[pid] for pid in node.inputs]
                # If a node has multiple parents, we concat them first as general fallback
                concat_var = f"concat_{var_name}"
                helper_nodes.append(f"""    # Merge branches for {var_name}
    node_concat_{var_name} = helper.make_node(
        "Concat",
        inputs={parents},
        outputs=["{concat_var}"],
        axis=1
    )
    onnx_nodes.append(node_concat_{var_name})""")
                input_arg = concat_var

            if op_type == "conv2d":
                filters = int(params.get("filters", 64))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                padding = params.get("padding", "valid")
                
                # Deduce padding value
                padding_str = padding.lower().strip() if isinstance(padding, str) else "valid"
                padding_val = kernel_size // 2 if padding_str == "same" else 0

                # Deduce input channels from input shape (channels_first index 1)
                in_channels = 3
                if node.input_shape and len(node.input_shape) >= 2:
                    in_channels = node.input_shape[1] if node.input_shape[1] is not None else 3

                # Weight initializer
                initializers.append(f"""    # Conv Weight Initializer for {var_name}
    w_shape_{var_name} = [{filters}, {in_channels}, {kernel_size}, {kernel_size}]
    w_val_{var_name} = np.random.randn(*w_shape_{var_name}).astype(np.float32) * 0.01
    w_tensor_{var_name} = helper.make_tensor(
        name="W_{var_name}",
        data_type=TensorProto.FLOAT,
        dims=w_shape_{var_name},
        vals=w_val_{var_name}.flatten()
    )
    onnx_initializers.append(w_tensor_{var_name})""")

                # Conv node
                conv_out = f"conv_out_{var_name}"
                helper_nodes.append(f"""    # Spatial Convolution: {var_name}
    node_{var_name} = helper.make_node(
        "Conv",
        inputs=["{input_arg}", "W_{var_name}"],
        outputs=["{conv_out}"],
        kernel_shape=[{kernel_size}, {kernel_size}],
        strides=[{stride}, {stride}],
        pads=[{padding_val}, {padding_val}, {padding_val}, {padding_val}]
    )
    onnx_nodes.append(node_{var_name})""")

                # Activation
                act_str = params.get("activation")
                if act_str and str(act_str).lower() != "none":
                    act_name = "Relu" if str(act_str).lower() == "relu" else str(act_str).capitalize()
                    helper_nodes.append(f"""    # Activation: {act_name} for {var_name}
    node_act_{var_name} = helper.make_node(
        "{act_name}",
        inputs=["{conv_out}"],
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_act_{var_name})""")
                else:
                    helper_nodes.append(f"""    # Identity mapping for {var_name}
    node_act_{var_name} = helper.make_node(
        "Identity",
        inputs=["{conv_out}"],
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_act_{var_name})""")

            elif op_type in (
                "maxpool", "maxpool2d", "maxpooling2d", "max_pooling2d",
                "avgpool", "avgpool2d", "avgpooling2d", "avg_pooling2d"
            ):
                pool_size = params.get("pool_size", 2)
                stride = params.get("stride", pool_size)
                
                pool_op = "MaxPool" if "max" in op_type else "AveragePool"
                helper_nodes.append(f"""    # Pooling downsampling: {var_name}
    node_{var_name} = helper.make_node(
        "{pool_op}",
        inputs=["{input_arg}"],
        outputs=["{var_name}"],
        kernel_shape=[{pool_size}, {pool_size}],
        strides=[{stride}, {stride}]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type in ("adaptivepool", "adaptiveavgpool", "adaptiveavgpool2d", "adaptivemaxpool2d"):
                # In ONNX, adaptive pooling is usually exported as GlobalAveragePool or GlobalMaxPool for size [1, 1]
                pool_op = "GlobalMaxPool" if "max" in op_type else "GlobalAveragePool"
                helper_nodes.append(f"""    # Global pooling: {var_name}
    node_{var_name} = helper.make_node(
        "{pool_op}",
        inputs=["{input_arg}"],
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type in ("convtranspose", "convtranspose2d"):
                filters = int(params.get("filters", 64))
                kernel_size = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                
                in_channels = 3
                if node.input_shape and len(node.input_shape) >= 2:
                    in_channels = node.input_shape[1] if node.input_shape[1] is not None else 3

                initializers.append(f"""    # ConvTranspose Weight Initializer for {var_name}
    w_shape_{var_name} = [{in_channels}, {filters}, {kernel_size}, {kernel_size}]
    w_val_{var_name} = np.random.randn(*w_shape_{var_name}).astype(np.float32) * 0.01
    w_tensor_{var_name} = helper.make_tensor(
        name="W_{var_name}",
        data_type=TensorProto.FLOAT,
        dims=w_shape_{var_name},
        vals=w_val_{var_name}.flatten()
    )
    onnx_initializers.append(w_tensor_{var_name})""")

                helper_nodes.append(f"""    # Spatial ConvTranspose: {var_name}
    node_{var_name} = helper.make_node(
        "ConvTranspose",
        inputs=["{input_arg}", "W_{var_name}"],
        outputs=["{var_name}"],
        kernel_shape=[{kernel_size}, {kernel_size}],
        strides=[{stride}, {stride}]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type == "flatten":
                helper_nodes.append(f"""    # Flatten tensor: {var_name}
    node_{var_name} = helper.make_node(
        "Flatten",
        inputs=["{input_arg}"],
        outputs=["{var_name}"],
        axis=1
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type in ("dense", "linear"):
                units = int(params.get("units") or params.get("out_features", 10))
                
                # Check input rank
                is_nd = node.input_shape and len(node.input_shape) > 2
                
                if is_nd:
                    in_features = node.input_shape[-1] if node.input_shape[-1] is not None else 100
                    # For MatMul, weight shape is [in_features, units], bias is [units]
                    initializers.append(f"""    # MatMul weights for {var_name}
    w_shape_{var_name} = [{in_features}, {units}]
    w_val_{var_name} = np.random.randn(*w_shape_{var_name}).astype(np.float32) * 0.05
    w_tensor_{var_name} = helper.make_tensor(
        name="W_{var_name}",
        data_type=TensorProto.FLOAT,
        dims=w_shape_{var_name},
        vals=w_val_{var_name}.flatten()
    )
    # Bias
    b_shape_{var_name} = [{units}]
    b_val_{var_name} = np.zeros(b_shape_{var_name}).astype(np.float32)
    b_tensor_{var_name} = helper.make_tensor(
        name="B_{var_name}",
        data_type=TensorProto.FLOAT,
        dims=b_shape_{var_name},
        vals=b_val_{var_name}.flatten()
    )
    onnx_initializers.extend([w_tensor_{var_name}, b_tensor_{var_name}])""")

                    matmul_out = f"matmul_out_{var_name}"
                    helper_nodes.append(f"""    # MatMul + Add projection for N-D: {var_name}
    node_matmul_{var_name} = helper.make_node(
        "MatMul",
        inputs=["{input_arg}", "W_{var_name}"],
        outputs=["{matmul_out}"]
    )
    node_{var_name} = helper.make_node(
        "Add",
        inputs=["{matmul_out}", "B_{var_name}"],
        outputs=["{var_name}"]
    )
    onnx_nodes.extend([node_matmul_{var_name}, node_{var_name}])""")
                else:
                    # Deduce 2D in_features
                    in_features = 100
                    if node.input_shape:
                        concrete_dims = [d for d in node.input_shape[1:] if d is not None]
                        if concrete_dims:
                            in_features = 1
                            for d in concrete_dims:
                                in_features *= d

                    initializers.append(f"""    # Gemm weights for {var_name}
    w_shape_{var_name} = [{units}, {in_features}]
    w_val_{var_name} = np.random.randn(*w_shape_{var_name}).astype(np.float32) * 0.05
    w_tensor_{var_name} = helper.make_tensor(
        name="W_{var_name}",
        data_type=TensorProto.FLOAT,
        dims=w_shape_{var_name},
        vals=w_val_{var_name}.flatten()
    )
    onnx_initializers.append(w_tensor_{var_name})""")

                    helper_nodes.append(f"""    # Gemm projection: {var_name}
    node_{var_name} = helper.make_node(
        "Gemm",
        inputs=["{input_arg}", "W_{var_name}"],
        outputs=["{var_name}"],
        transB=1
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type in ("batchnorm", "batchnorm2d"):
                in_channels = 3
                if node.input_shape and len(node.input_shape) >= 2:
                    in_channels = node.input_shape[1] if node.input_shape[1] is not None else 3

                initializers.append(f"""    # BatchNorm params for {var_name}
    scale_val_{var_name} = np.ones([{in_channels}]).astype(np.float32)
    bias_val_{var_name} = np.zeros([{in_channels}]).astype(np.float32)
    mean_val_{var_name} = np.zeros([{in_channels}]).astype(np.float32)
    var_val_{var_name} = np.ones([{in_channels}]).astype(np.float32)
    
    scale_tensor_{var_name} = helper.make_tensor("scale_{var_name}", TensorProto.FLOAT, [{in_channels}], scale_val_{var_name})
    bias_tensor_{var_name} = helper.make_tensor("bias_{var_name}", TensorProto.FLOAT, [{in_channels}], bias_val_{var_name})
    mean_tensor_{var_name} = helper.make_tensor("mean_{var_name}", TensorProto.FLOAT, [{in_channels}], mean_val_{var_name})
    var_tensor_{var_name} = helper.make_tensor("var_{var_name}", TensorProto.FLOAT, [{in_channels}], var_val_{var_name})
    
    onnx_initializers.extend([scale_tensor_{var_name}, bias_tensor_{var_name}, mean_tensor_{var_name}, var_tensor_{var_name}])""")

                helper_nodes.append(f"""    # Batch Normalization: {var_name}
    node_{var_name} = helper.make_node(
        "BatchNormalization",
        inputs=["{input_arg}", "scale_{var_name}", "bias_{var_name}", "mean_{var_name}", "var_{var_name}"],
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type == "dropout":
                # ONNX opset 13: Dropout outputs (output, mask). ratio is an optional input, not an attribute.
                # For inference export we emit a single-output Dropout node (training=False, ratio passed as input).
                # Simplest cross-opset approach: use Identity to preserve the data flow in inference mode.
                helper_nodes.append(f"""    # Dropout (inference-mode Identity): {var_name}
    node_{var_name} = helper.make_node(
        "Identity",
        inputs=["{input_arg}"],
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type in ("multiheadattention", "mha"):
                parents = [node_vars[pid] for pid in node.inputs]
                if len(parents) == 3:
                    q, k, v = parents[0], parents[1], parents[2]
                elif len(parents) == 2:
                    q, k, v = parents[0], parents[1], parents[1]
                else:
                    q, k, v = input_arg, input_arg, input_arg
                
                kt_var = f"kt_{var_name}"
                helper_nodes.append(f"""    # Attention Graph Routing - Transpose Key for {var_name}
    node_kt_{var_name} = helper.make_node(
        "Transpose",
        inputs=["{k}"],
        outputs=["{kt_var}"],
        perm=[0, 2, 1]
    )
    onnx_nodes.append(node_kt_{var_name})""")

                qk_var = f"qk_{var_name}"
                helper_nodes.append(f"""    # Attention Graph Routing - Q x K^T MatMul for {var_name}
    node_qk_{var_name} = helper.make_node(
        "MatMul",
        inputs=["{q}", "{kt_var}"],
        outputs=["{qk_var}"]
    )
    onnx_nodes.append(node_qk_{var_name})""")

                softmax_var = f"softmax_{var_name}"
                helper_nodes.append(f"""    # Attention Graph Routing - Softmax for {var_name}
    node_softmax_{var_name} = helper.make_node(
        "Softmax",
        inputs=["{qk_var}"],
        outputs=["{softmax_var}"],
        axis=-1
    )
    onnx_nodes.append(node_softmax_{var_name})""")

                helper_nodes.append(f"""    # Attention Graph Routing - Attention Weight x V MatMul for {var_name}
    node_attn_out_{var_name} = helper.make_node(
        "MatMul",
        inputs=["{softmax_var}", "{v}"],
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_attn_out_{var_name})""")

            elif op_type == "add":
                parents = [node_vars[pid] for pid in node.inputs]
                helper_nodes.append(f"""    # Elementwise Add: {var_name}
    node_{var_name} = helper.make_node(
        "Add",
        inputs={parents},
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type == "subtract":
                parents = [node_vars[pid] for pid in node.inputs]
                helper_nodes.append(f"""    # Elementwise Subtract: {var_name}
    node_{var_name} = helper.make_node(
        "Sub",
        inputs={parents},
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type == "maximum":
                parents = [node_vars[pid] for pid in node.inputs]
                helper_nodes.append(f"""    # Elementwise Maximum: {var_name}
    node_{var_name} = helper.make_node(
        "Max",
        inputs={parents},
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type == "minimum":
                parents = [node_vars[pid] for pid in node.inputs]
                helper_nodes.append(f"""    # Elementwise Minimum: {var_name}
    node_{var_name} = helper.make_node(
        "Min",
        inputs={parents},
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type == "multiply":
                parents = [node_vars[pid] for pid in node.inputs]
                helper_nodes.append(f"""    # Elementwise Multiply: {var_name}
    node_{var_name} = helper.make_node(
        "Mul",
        inputs={parents},
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type == "concatenate":
                axis = int(params.get("axis", 1))
                parents = [node_vars[pid] for pid in node.inputs]
                helper_nodes.append(f"""    # Concat: {var_name}
    node_{var_name} = helper.make_node(
        "Concat",
        inputs={parents},
        outputs=["{var_name}"],
        axis={axis}
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type == "reshape":
                target_shape = params.get("shape") or params.get("target_shape", [-1])
                cleaned_shape = [dim if dim is not None else -1 for dim in target_shape]
                
                initializers.append(f"""    # Reshape target shape initializer for {var_name}
    shape_tensor_{var_name} = helper.make_tensor(
        name="shape_{var_name}",
        data_type=TensorProto.INT64,
        dims=[{len(cleaned_shape)}],
        vals={cleaned_shape}
    )
    onnx_initializers.append(shape_tensor_{var_name})""")

                helper_nodes.append(f"""    # Reshape: {var_name}
    node_{var_name} = helper.make_node(
        "Reshape",
        inputs=["{input_arg}", "shape_{var_name}"],
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_{var_name})""")

            elif op_type in ("permute", "transpose"):
                axes = params.get("axes") or params.get("dims", [0, 2, 1, 3])
                helper_nodes.append(f"""    # Transpose/Permute: {var_name}
    node_{var_name} = helper.make_node(
        "Transpose",
        inputs=["{input_arg}"],
        outputs=["{var_name}"],
        perm={axes}
    )
    onnx_nodes.append(node_{var_name})""")

            else:
                # Identity fallback
                helper_nodes.append(f"""    # Identity fallback: {var_name}
    node_{var_name} = helper.make_node(
        "Identity",
        inputs=["{input_arg}"],
        outputs=["{var_name}"]
    )
    onnx_nodes.append(node_{var_name})""")

        # Calculate final returns (leaves)
        final_leaves_nodes = [node for node in sorted_nodes if outgoing_count[node.id] == 0]
        final_leaves = []
        for leaf in final_leaves_nodes:
            leaf_var = node_vars[leaf.id]
            # Replace dynamic batch None/dynamic dims with 1 for ONNX make_tensor_value_info
            if leaf.output_shape:
                leaf_shape = [d if d is not None else 1 for d in leaf.output_shape]
            else:
                leaf_shape = [1, 10]
            
            final_leaves.append({
                "name": leaf_var,
                "shape_dims": str(leaf_shape)
            })

        # Render template
        rendered_code = template.render(
            project_name=ir_graph.project_name,
            framework="ONNX",
            initializers=initializers,
            helper_nodes=helper_nodes,
            input_dims=str(input_dims),
            final_leaves=final_leaves
        )
        
        return rendered_code
