import math
import uuid
from typing import List, Dict, Any
from app.models.node import Node
from app.models.edge import Edge

class ShapeInferenceService:
    @staticmethod
    def parse_int_or_tuple(val: Any) -> tuple[int, int]:
        """Convert a single integer or a list/tuple of 1 or 2 ints into a 2-tuple of ints."""
        if isinstance(val, int):
            return (val, val)
        if isinstance(val, (list, tuple)):
            if len(val) == 1:
                return (val[0], val[0])
            elif len(val) >= 2:
                return (val[0], val[1])
        # If it's a string, try parsing it
        if isinstance(val, str):
            try:
                cleaned = val.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
                parts = [int(p.strip()) for p in cleaned.split(",")]
                if len(parts) == 1:
                    return (parts[0], parts[0])
                elif len(parts) >= 2:
                    return (parts[0], parts[1])
            except ValueError:
                pass
        raise ValueError(f"Invalid parameter format: {val}. Expected integer or list/tuple.")

    @staticmethod
    def broadcast_shapes(shape_a: List[Any], shape_b: List[Any]) -> List[Any]:
        """Broadcast two shapes according to standard NumPy/PyTorch rules.
        Supports symbolic/dynamic None values.
        """
        if not shape_a or not shape_b:
            raise ValueError("Cannot broadcast empty shapes.")

        len_a, len_b = len(shape_a), len(shape_b)
        max_len = max(len_a, len_b)

        # Pad shorter shapes with 1s on the left
        pad_a = [1] * (max_len - len_a) + list(shape_a)
        pad_b = [1] * (max_len - len_b) + list(shape_b)

        out_shape = []
        for idx in range(max_len):
            dim_a = pad_a[idx]
            dim_b = pad_b[idx]

            if dim_a == dim_b:
                out_shape.append(dim_a)
            elif dim_a is None or dim_b is None:
                # Dynamic/symbolic dimension: propagate None
                out_shape.append(None)
            elif dim_a == 1:
                out_shape.append(dim_b)
            elif dim_b == 1:
                out_shape.append(dim_a)
            else:
                raise ValueError(f"Shape mismatch: dimensions {dim_a} and {dim_b} are incompatible for broadcasting.")
        return out_shape

    @staticmethod
    def infer_node_shape(node: Node, input_shapes: List[List[Any]]) -> List[Any]:
        """Calculates the output shape for a single node based on its type, config, and incoming shapes list.
        Fully supports symbolic dynamic shapes and broadcasting rules.
        """
        if input_shapes and not isinstance(input_shapes[0], list):
            input_shapes = [input_shapes]
            
        node_type = node.type.lower()
        config = node.config or {}

        if node_type == "input":
            shape = config.get("shape") or config.get("input_shape")
            if not shape:
                raise ValueError("Input layer must have a 'shape' defined in its config (e.g., [None, 3, 224, 224]).")
            if not isinstance(shape, list):
                raise ValueError("Input shape must be a list of dimensions.")
            return shape

        if not input_shapes or not any(input_shapes):
            raise ValueError(f"Input shape to layer '{node.label}' is missing or empty.")

        first_input = input_shapes[0]
        batch_size = first_input[0]

        if node_type == "conv2d":
            if len(first_input) != 4:
                raise ValueError(
                    f"Conv2D layer '{node.label}' requires 4D input [Batch, Channels, Height, Width]. "
                    f"Received shape: {first_input}"
                )
            
            in_channels, in_h, in_w = first_input[1], first_input[2], first_input[3]
            filters = int(config.get("filters", 32))

            # Dynamic Spatial check: if height or width is dynamic (None), propagate None
            if in_h is None or in_w is None:
                return [batch_size, filters, None, None]

            kernel_size = ShapeInferenceService.parse_int_or_tuple(config.get("kernel_size", 3))
            stride = ShapeInferenceService.parse_int_or_tuple(config.get("stride", 1))
            padding_config = config.get("padding", "valid")

            kh, kw = kernel_size
            sh, sw = stride

            ph, pw = 0, 0
            if isinstance(padding_config, int):
                ph, pw = padding_config, padding_config
            elif isinstance(padding_config, (list, tuple)):
                ph, pw = ShapeInferenceService.parse_int_or_tuple(padding_config)
            elif isinstance(padding_config, str):
                padding_str = padding_config.lower().strip()
                if padding_str == "same":
                    out_h = math.ceil(in_h / sh)
                    out_w = math.ceil(in_w / sw)
                    return [batch_size, filters, out_h, out_w]
                elif padding_str == "valid":
                    ph, pw = 0, 0
                else:
                    raise ValueError(f"Unsupported padding string: '{padding_config}' for Conv2D.")

            out_h = math.floor((in_h + 2 * ph - kh) / sh) + 1
            out_w = math.floor((in_w + 2 * pw - kw) / sw) + 1

            if out_h <= 0 or out_w <= 0:
                raise ValueError(
                    f"Conv2D layer '{node.label}' configuration results in negative or zero spatial dimensions. "
                    f"Input shape: {first_input}, kernel_size: {kernel_size}, stride: {stride}."
                )

            return [batch_size, filters, out_h, out_w]

        elif node_type in ("maxpool2d", "avgpool", "avgpool2d"):
            if len(first_input) != 4:
                raise ValueError(
                    f"Pooling layer '{node.label}' requires 4D input [Batch, Channels, Height, Width]. "
                    f"Received shape: {first_input}"
                )
            
            in_channels, in_h, in_w = first_input[1], first_input[2], first_input[3]
            
            # Dynamic Spatial check: propagate None
            if in_h is None or in_w is None:
                return [batch_size, in_channels, None, None]

            pool_size = ShapeInferenceService.parse_int_or_tuple(config.get("pool_size", 2))
            stride = ShapeInferenceService.parse_int_or_tuple(config.get("stride", pool_size))
            padding_config = config.get("padding", 0)

            kh, kw = pool_size
            sh, sw = stride

            ph, pw = 0, 0
            if isinstance(padding_config, int):
                ph, pw = padding_config, padding_config
            elif isinstance(padding_config, (list, tuple)):
                ph, pw = ShapeInferenceService.parse_int_or_tuple(padding_config)
            elif isinstance(padding_config, str):
                if padding_config.lower().strip() == "same":
                    out_h = math.ceil(in_h / sh)
                    out_w = math.ceil(in_w / sw)
                    return [batch_size, in_channels, out_h, out_w]

            out_h = math.floor((in_h + 2 * ph - kh) / sh) + 1
            out_w = math.floor((in_w + 2 * pw - kw) / sw) + 1

            if out_h <= 0 or out_w <= 0:
                raise ValueError(
                    f"Pooling layer '{node.label}' configuration results in negative or zero spatial dimensions. "
                    f"Input shape: {first_input}, pool_size: {pool_size}, stride: {stride}."
                )

            return [batch_size, in_channels, out_h, out_w]

        elif node_type in ("adaptivepool", "adaptiveavgpool", "adaptiveavgpool2d", "adaptivemaxpool2d"):
            if len(first_input) != 4:
                raise ValueError(f"Adaptive pooling layer '{node.label}' requires 4D input. Received shape: {first_input}")
            
            in_channels = first_input[1]
            output_size = ShapeInferenceService.parse_int_or_tuple(config.get("output_size", 1))
            return [batch_size, in_channels, output_size[0], output_size[1]]

        elif node_type in ("convtranspose", "convtranspose2d"):
            if len(first_input) != 4:
                raise ValueError(f"ConvTranspose2D layer '{node.label}' requires 4D input. Received shape: {first_input}")
            
            in_channels, in_h, in_w = first_input[1], first_input[2], first_input[3]
            filters = int(config.get("filters", 32))

            # Dynamic Spatial check: propagate None
            if in_h is None or in_w is None:
                return [batch_size, filters, None, None]

            kernel_size = ShapeInferenceService.parse_int_or_tuple(config.get("kernel_size", 3))
            stride = ShapeInferenceService.parse_int_or_tuple(config.get("stride", 1))
            padding = ShapeInferenceService.parse_int_or_tuple(config.get("padding", 0))
            output_padding = ShapeInferenceService.parse_int_or_tuple(config.get("output_padding", 0))

            kh, kw = kernel_size
            sh, sw = stride
            ph, pw = padding
            oph, opw = output_padding

            out_h = (in_h - 1) * sh - 2 * ph + kh + oph
            out_w = (in_w - 1) * sw - 2 * pw + kw + opw

            return [batch_size, filters, out_h, out_w]

        elif node_type == "flatten":
            if len(first_input) < 2:
                raise ValueError(f"Flatten layer '{node.label}' requires input dimension >= 2. Received shape: {first_input}")
                
            multiplier = 1
            for dim in first_input[1:]:
                if dim is None:
                    # In dynamic sequences or resolutions, return dynamic flat dimension
                    return [batch_size, None]
                multiplier *= int(dim)

            return [batch_size, multiplier]

        elif node_type in ("dense", "linear"):
            if len(first_input) != 2:
                raise ValueError(
                    f"Dense/Linear layer '{node.label}' requires 2D input [Batch, Features]. "
                    f"Received shape: {first_input}."
                )
            
            units = int(config.get("units") or config.get("out_features", 10))
            return [batch_size, units]

        elif node_type in ("batchnorm", "batchnorm2d", "layernorm", "positional_encoding"):
            return first_input

        elif node_type in ("lstm", "gru", "rnn"):
            if len(first_input) != 3:
                raise ValueError(f"RNN layer '{node.label}' requires 3D input [Batch, Seq_Len, Features]. Received shape: {first_input}")
            
            seq_len = first_input[1]
            hidden_size = int(config.get("hidden_size") or config.get("units", 64))
            return_sequences = bool(config.get("return_sequences", False))

            if return_sequences:
                return [batch_size, seq_len, hidden_size]
            else:
                return [batch_size, hidden_size]

        elif node_type == "bidirectional":
            if len(first_input) != 3:
                raise ValueError(f"Bidirectional RNN layer '{node.label}' requires 3D input [Batch, Seq_Len, Features]. Received shape: {first_input}")
            
            seq_len = first_input[1]
            hidden_size = int(config.get("hidden_size") or config.get("units", 64))
            return_sequences = bool(config.get("return_sequences", False))

            if return_sequences:
                return [batch_size, seq_len, 2 * hidden_size]
            else:
                return [batch_size, 2 * hidden_size]

        elif node_type == "embedding":
            if len(first_input) != 2:
                raise ValueError(f"Embedding layer '{node.label}' requires 2D input [Batch, Seq_Len]. Received shape: {first_input}")
            
            seq_len = first_input[1]
            embed_dim = int(config.get("embedding_dim") or config.get("output_dim", 128))
            return [batch_size, seq_len, embed_dim]

        elif node_type in ("multiheadattention", "mha"):
            if len(first_input) != 3:
                raise ValueError(f"Attention layer '{node.label}' requires 3D inputs. Received shape: {first_input}")
            return first_input

        elif node_type in ("add", "multiply"):
            if len(input_shapes) < 2:
                raise ValueError(f"Operation layer '{node.label}' requires at least 2 incoming connections. Received {len(input_shapes)}")
                
            out_shape = input_shapes[0]
            for idx, shape in enumerate(input_shapes[1:]):
                try:
                    out_shape = ShapeInferenceService.broadcast_shapes(out_shape, shape)
                except ValueError as e:
                    raise ValueError(
                        f"Operation layer '{node.label}' broadcasting failed. "
                        f"Index 0 base: {input_shapes[0]} vs Index {idx + 1}: {shape}. Detail: {e}"
                    )
            return out_shape

        elif node_type == "concatenate":
            if len(input_shapes) < 2:
                raise ValueError(f"Concatenate layer '{node.label}' requires at least 2 incoming connections. Received {len(input_shapes)}")
            
            axis = int(config.get("axis", 1))
            base_shape = input_shapes[0]
            num_dims = len(base_shape)
            
            if axis < 0:
                axis = num_dims + axis
                
            if axis >= num_dims or axis < 0:
                raise ValueError(f"Concatenate layer '{node.label}' axis {axis} is out of bounds for shape {base_shape}")

            concat_dim_total = base_shape[axis]
            
            for idx, shape in enumerate(input_shapes[1:]):
                if len(shape) != num_dims:
                    raise ValueError(
                        f"Concatenate layer '{node.label}' inputs must have matching number of dimensions. "
                        f"Index 0: {base_shape} vs Index {idx + 1}: {shape}"
                    )
                for dim_idx in range(num_dims):
                    if dim_idx != axis:
                        if shape[dim_idx] != base_shape[dim_idx]:
                            raise ValueError(
                                f"Concatenate layer '{node.label}' inputs must match along non-concatenation axes. "
                                f"Mismatch at axis {dim_idx}: Index 0: {base_shape} vs Index {idx + 1}: {shape}"
                            )
                
                if base_shape[axis] is not None and shape[axis] is not None:
                    concat_dim_total += shape[axis]
                else:
                    concat_dim_total = None

            output_shape = list(base_shape)
            output_shape[axis] = concat_dim_total
            return output_shape

        elif node_type == "reshape":
            target_shape = config.get("shape") or config.get("target_shape")
            if not target_shape or not isinstance(target_shape, list):
                raise ValueError(f"Reshape layer '{node.label}' requires a target 'shape' configuration (e.g. [-1, 512]).")

            resolved_shape = list(target_shape)
            if resolved_shape[0] is not None and resolved_shape[0] != "None":
                resolved_shape.insert(0, batch_size)
            else:
                resolved_shape[0] = batch_size

            input_vol = 1
            dynamic_idx = -1
            
            for idx, dim in enumerate(resolved_shape[1:]):
                actual_idx = idx + 1
                if dim == -1:
                    if dynamic_idx != -1:
                        raise ValueError(f"Reshape layer '{node.label}' target shape cannot have more than one dynamic dimension (-1).")
                    dynamic_idx = actual_idx
                elif dim is not None:
                    input_vol *= int(dim)

            incoming_vol = 1
            for dim in first_input[1:]:
                if dim is None:
                    # In dynamic sequence or dynamic resolution pipelines, return dynamic dimension
                    # E.g. [None, None, 512]
                    resolved_shape[1] = None
                    return resolved_shape
                incoming_vol *= int(dim)

            if dynamic_idx != -1:
                if incoming_vol % input_vol != 0:
                    raise ValueError(
                        f"Reshape layer '{node.label}' target volume mismatch. "
                        f"Incoming elements: {incoming_vol} vs Target base: {input_vol}."
                    )
                resolved_shape[dynamic_idx] = incoming_vol // input_vol
            else:
                target_vol = 1
                for dim in resolved_shape[1:]:
                    if dim is not None:
                        target_vol *= int(dim)
                if target_vol != incoming_vol:
                    raise ValueError(
                        f"Reshape layer '{node.label}' target shape volume {target_vol} "
                        f"does not match incoming shape volume {incoming_vol}."
                    )

            return resolved_shape

        elif node_type in ("permute", "transpose"):
            axes = config.get("axes") or config.get("dims")
            if not axes or not isinstance(axes, list):
                raise ValueError(f"Permute layer '{node.label}' requires a list of 'axes' permutation mapping.")
            
            if len(axes) == len(first_input) - 1:
                axes = [0] + [a + 1 for a in axes]
                
            if len(axes) != len(first_input):
                raise ValueError(f"Permute layer '{node.label}' axes length {len(axes)} does not match input dimensions {len(first_input)}")

            return [first_input[axis] for axis in axes]

        return first_input

    @classmethod
    def run_shape_inference(cls, sorted_nodes: List[Node], edges: List[Edge]) -> None:
        """Traverses the sorted graph DAG and updates the input_shape and output_shape columns of each Node in place.
        Supports multi-input nodes like Add and Concatenate.
        """
        if not sorted_nodes:
            return

        # Mapping node ID -> resolved output shape
        node_output_shapes: Dict[uuid.UUID, List[Any]] = {}

        # Build adjacency maps
        incoming_edges_map: Dict[uuid.UUID, List[Edge]] = {n.id: [] for n in sorted_nodes}
        for edge in edges:
            if edge.to_node_id in incoming_edges_map:
                incoming_edges_map[edge.to_node_id].append(edge)

        for node in sorted_nodes:
            if node.type.lower() == "input":
                node.input_shape = None
                out_shape = cls.infer_node_shape(node, [[None]])
                node.output_shape = out_shape
                node_output_shapes[node.id] = out_shape
            else:
                incoming = incoming_edges_map[node.id]
                if not incoming:
                    raise ValueError(f"Layer '{node.label}' has no incoming connections and is not an 'Input' layer.")

                # Gather output shapes from all predecessor nodes
                parent_shapes = []
                for edge in incoming:
                    parent_shape = node_output_shapes.get(edge.from_node_id)
                    if not parent_shape:
                        raise ValueError(
                            f"Input shape from parent '{edge.from_node_id}' for layer '{node.label}' was not yet computed. "
                            "Verify topological order or cyclic connections."
                        )
                    parent_shapes.append(parent_shape)
                    
                    # Update shape metadata on the edge
                    edge.input_shape = parent_shape

                # If single input, store flat list. Otherwise, store as list of shapes
                if len(parent_shapes) == 1:
                    node.input_shape = parent_shapes[0]
                else:
                    node.input_shape = parent_shapes

                # Run math inference
                out_shape = cls.infer_node_shape(node, parent_shapes)
                node.output_shape = out_shape
                node_output_shapes[node.id] = out_shape

                # Update outputs on the edge
                for edge in incoming:
                    edge.output_shape = out_shape
