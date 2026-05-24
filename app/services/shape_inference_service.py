import math
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
                # Remove brackets/parentheses and split
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
    def infer_node_shape(node: Node, input_shape: List[Any]) -> List[Any]:
        """Calculates the output shape for a single node based on its type, config, and input shape."""
        node_type = node.type.lower()
        config = node.config or {}

        # Ensure input_shape is valid
        if not input_shape:
            raise ValueError(f"Input shape to layer '{node.label}' is missing or empty.")

        # Batch size is typically index 0 (e.g. None or int)
        batch_size = input_shape[0]

        if node_type == "input":
            # Input layer output shape is directly defined in its config
            shape = config.get("shape") or config.get("input_shape")
            if not shape:
                raise ValueError("Input layer must have a 'shape' defined in its config (e.g., [None, 3, 224, 224]).")
            if not isinstance(shape, list):
                raise ValueError("Input shape must be a list of dimensions.")
            return shape

        elif node_type == "conv2d":
            # Expected input: [Batch, Channels, Height, Width]
            if len(input_shape) != 4:
                raise ValueError(
                    f"Conv2D layer '{node.label}' requires 4D input [Batch, Channels, Height, Width]. "
                    f"Received shape: {input_shape}"
                )
            
            in_channels, in_h, in_w = input_shape[1], input_shape[2], input_shape[3]
            
            # Read parameters from config
            filters = int(config.get("filters", 32))
            kernel_size = ShapeInferenceService.parse_int_or_tuple(config.get("kernel_size", 3))
            stride = ShapeInferenceService.parse_int_or_tuple(config.get("stride", 1))
            padding_config = config.get("padding", "valid")

            kh, kw = kernel_size
            sh, sw = stride

            # Calculate padding values
            ph, pw = 0, 0
            if isinstance(padding_config, int):
                ph, pw = padding_config, padding_config
            elif isinstance(padding_config, (list, tuple)):
                ph, pw = ShapeInferenceService.parse_int_or_tuple(padding_config)
            elif isinstance(padding_config, str):
                padding_str = padding_config.lower().strip()
                if padding_str == "same":
                    # Same padding output is ceil(input/stride)
                    out_h = math.ceil(in_h / sh)
                    out_w = math.ceil(in_w / sw)
                    return [batch_size, filters, out_h, out_w]
                elif padding_str == "valid":
                    ph, pw = 0, 0
                else:
                    raise ValueError(f"Unsupported padding string: '{padding_config}' for Conv2D.")

            # Standard formula: out = floor((in + 2P - K)/S) + 1
            out_h = math.floor((in_h + 2 * ph - kh) / sh) + 1
            out_w = math.floor((in_w + 2 * pw - kw) / sw) + 1

            if out_h <= 0 or out_w <= 0:
                raise ValueError(
                    f"Conv2D layer '{node.label}' configuration results in negative or zero spatial dimensions. "
                    f"Input shape: {input_shape}, kernel_size: {kernel_size}, stride: {stride}, padding: {padding_config}."
                )

            return [batch_size, filters, out_h, out_w]

        elif node_type == "maxpool2d":
            # Expected input: [Batch, Channels, Height, Width]
            if len(input_shape) != 4:
                raise ValueError(
                    f"MaxPool2D layer '{node.label}' requires 4D input [Batch, Channels, Height, Width]. "
                    f"Received shape: {input_shape}"
                )
            
            in_channels, in_h, in_w = input_shape[1], input_shape[2], input_shape[3]
            
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
                # Default valid is 0

            out_h = math.floor((in_h + 2 * ph - kh) / sh) + 1
            out_w = math.floor((in_w + 2 * pw - kw) / sw) + 1

            if out_h <= 0 or out_w <= 0:
                raise ValueError(
                    f"MaxPool2D layer '{node.label}' configuration results in negative or zero spatial dimensions. "
                    f"Input shape: {input_shape}, pool_size: {pool_size}, stride: {stride}."
                )

            return [batch_size, in_channels, out_h, out_w]

        elif node_type == "flatten":
            # Flattens spatial dimensions into a single feature vector
            # Input: [Batch, Channels, Height, Width] -> Output: [Batch, Channels * Height * Width]
            # Also supports flattening 3D/2D inputs if any
            if len(input_shape) < 2:
                raise ValueError(f"Flatten layer '{node.label}' requires input dimension >= 2. Received shape: {input_shape}")
                
            multiplier = 1
            for dim in input_shape[1:]:
                if dim is None:
                    raise ValueError(f"Flatten layer '{node.label}' cannot flatten dynamic dimensions (None). Input shape: {input_shape}")
                multiplier *= int(dim)

            return [batch_size, multiplier]

        elif node_type in ("dense", "linear"):
            # Expected input: [Batch, in_features]
            if len(input_shape) != 2:
                raise ValueError(
                    f"Dense/Linear layer '{node.label}' requires 2D input [Batch, Features]. "
                    f"Received shape: {input_shape}. (Hint: Insert a 'Flatten' layer before Dense if using Conv2D output)"
                )
            
            units = int(config.get("units") or config.get("out_features", 10))
            return [batch_size, units]

        # Unknown layer: simply pass-through shape but log warning/keep same shape
        return input_shape

    @classmethod
    def run_shape_inference(cls, sorted_nodes: List[Node], edges: List[Edge]) -> None:
        """Traverses the sorted graph DAG and updates the input_shape and output_shape columns of each Node in place.
        Raises ValueError if there is an incompatibility or math error.
        """
        if not sorted_nodes:
            return

        # Create mapping to quickly retrieve source node output shapes
        node_output_shapes: Dict[uuid.UUID, List[Any]] = {}
        node_map = {n.id: n for n in sorted_nodes}

        # Build adjacency maps to find inputs of each node
        incoming_edges_map: Dict[uuid.UUID, List[Edge]] = {n.id: [] for n in sorted_nodes}
        for edge in edges:
            if edge.to_node_id in incoming_edges_map:
                incoming_edges_map[edge.to_node_id].append(edge)

        for node in sorted_nodes:
            if node.type.lower() == "input":
                # Input node doesn't have ancestors, get starting shape from its config
                node.input_shape = None
                out_shape = cls.infer_node_shape(node, [None]) # config dictates output
                node.output_shape = out_shape
                node_output_shapes[node.id] = out_shape
            else:
                # Retrieve input shape from incoming edges
                incoming = incoming_edges_map[node.id]
                if not incoming:
                    raise ValueError(f"Layer '{node.label}' has no incoming connections and is not an 'Input' layer.")
                
                # For Phase 1 sequential models, we assume single input path
                # Use the output shape of the first incoming edge's source node
                source_edge = incoming[0]
                parent_shape = node_output_shapes.get(source_edge.from_node_id)
                
                if not parent_shape:
                    raise ValueError(f"Input shape from parent for layer '{node.label}' was not computed.")
                
                node.input_shape = parent_shape
                out_shape = cls.infer_node_shape(node, parent_shape)
                node.output_shape = out_shape
                node_output_shapes[node.id] = out_shape
                
                # Write shape back to the edges too for database updates
                source_edge.input_shape = parent_shape
                source_edge.output_shape = out_shape
