import uuid
from typing import List, Dict, Set
from app.models.node import Node
from app.models.edge import Edge

class ValidationService:
    @staticmethod
    def topological_sort(nodes: List[Node], edges: List[Edge]) -> List[Node]:
        """Perform topological sort on the nodes based on edges to detect cycles and execution order.
        Raises ValueError if a cycle is detected.
        """
        node_ids = {node.id for node in nodes}
        adj: Dict[uuid.UUID, List[uuid.UUID]] = {nid: [] for nid in node_ids}
        in_degree: Dict[uuid.UUID, int] = {nid: 0 for nid in node_ids}

        # Build graph representation
        for edge in edges:
            # Skip edges pointing outside our active nodes list
            if edge.from_node_id in node_ids and edge.to_node_id in node_ids:
                adj[edge.from_node_id].append(edge.to_node_id)
                in_degree[edge.to_node_id] += 1

        # Queue of nodes with no incoming connections (sources)
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_ids: List[uuid.UUID] = []

        while queue:
            # Maintain stable sorting order by popping the first element
            u = queue.pop(0)
            sorted_ids.append(u)

            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(sorted_ids) != len(nodes):
            raise ValueError("Invalid architecture: Graph contains cycles (loops are not allowed).")

        # Map sorted UUIDs back to node objects
        node_map = {node.id: node for node in nodes}
        return [node_map[nid] for nid in sorted_ids]

    @staticmethod
    def validate_graph(nodes: List[Node], edges: List[Edge]) -> List[Node]:
        """Validates the neural network DAG.
        Checks:
        1. Graph is not empty
        2. Exactly one Input node exists
        3. No cycles exist (Topological Sort checks this)
        4. Detects disconnected graphs (reachability from Input node)
        
        Returns the topologically sorted list of nodes if valid.
        """
        if not nodes:
            raise ValueError("Invalid architecture: Graph is empty.")

        # Find the input layer
        input_nodes = [n for n in nodes if n.type.lower() == "input"]
        if len(input_nodes) == 0:
            raise ValueError("Invalid architecture: Missing 'Input' layer.")
        if len(input_nodes) > 1:
            raise ValueError("Invalid architecture: Multiple 'Input' layers detected. Only one is allowed.")
        
        input_node = input_nodes[0]

        # 1. Sort the graph (checks for cycles)
        sorted_nodes = ValidationService.topological_sort(nodes, edges)

        # 2. Check reachability from Input node (detect disconnected graph parts)
        node_ids = {node.id for node in nodes}
        adj: Dict[uuid.UUID, List[uuid.UUID]] = {nid: [] for nid in node_ids}
        for edge in edges:
            if edge.from_node_id in node_ids and edge.to_node_id in node_ids:
                adj[edge.from_node_id].append(edge.to_node_id)

        visited: Set[uuid.UUID] = set()
        queue = [input_node.id]

        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        # If any node is not reachable from the input node, we have disconnected components
        unreachable = node_ids - visited
        if unreachable:
            unreachable_labels = [n.label for n in nodes if n.id in unreachable]
            raise ValueError(
                f"Invalid architecture: Disconnected layers detected {unreachable_labels}. "
                "All layers must connect to the path starting from the 'Input' layer."
            )

        return sorted_nodes

    @staticmethod
    def validate_semantics(nodes: List[Node], edges: List[Edge]) -> List[str]:
        """Performs semantic validation on the nodes in the graph.
        Checks:
        1. Invalid tensor ranks for layers (e.g. Conv2D expects 4D, Dense expects 2D, RNN expects 3D).
        2. Multi-Head Attention heads divisibility rules.
        3. Reshape operations volume and dynamic dimension constraints.
        4. Illegal broadcasting for element-wise merges (Add, Multiply).
        
        Returns a list of validation error messages. An empty list means valid semantics.
        """
        errors: List[str] = []
        
        # Build adjacency for multi-parent inputs
        node_map = {node.id: node for node in nodes}
        incoming_edges: Dict[uuid.UUID, List[Edge]] = {nid: [] for nid in node_map}
        for edge in edges:
            if edge.to_node_id in incoming_edges:
                incoming_edges[edge.to_node_id].append(edge)

        for node in nodes:
            node_type = node.type.lower()
            config = node.config or {}
            
            # Resolve input shape from node or preceding edge
            input_shape = node.input_shape
            parents = incoming_edges[node.id]
            
            # 0. Incoming Connection Counts Validation
            if edges:
                parent_count = len(parents)
                if node_type in ("add", "multiply", "subtract", "maximum", "minimum", "concatenate"):
                    if parent_count < 2:
                        errors.append(
                            f"Merge layer '{node.label}' ({node.type}) requires at least 2 incoming connections. "
                            f"Received: {parent_count}."
                        )
                elif node_type in ("multiheadattention", "mha"):
                    if parent_count < 1 or parent_count > 3:
                        errors.append(
                            f"Attention layer '{node.label}' ({node.type}) requires between 1 and 3 incoming connections. "
                            f"Received: {parent_count}."
                        )
                elif node_type != "input":
                    # Single-input layers
                    if parent_count != 1:
                        errors.append(
                            f"Layer '{node.label}' ({node.type}) requires exactly 1 incoming connection. "
                            f"Received: {parent_count}. If you want to connect multiple branches, "
                            f"use a merge layer (like 'Add' or 'Concatenate') first."
                        )
            
            # Helper to get preceding shapes
            parent_shapes = []
            for p in parents:
                parent_node = node_map.get(p.from_node_id)
                if parent_node and parent_node.output_shape:
                    parent_shapes.append(parent_node.output_shape)
                    
            if not input_shape and parent_shapes:
                input_shape = parent_shapes[0]

            # 1. Invalid Tensor Ranks
            if input_shape:
                rank = len(input_shape)
                if node_type in ("conv2d", "maxpool2d", "avgpool", "avgpool2d", "batchnorm", "batchnorm2d", "adaptivepool", "adaptiveavgpool", "adaptiveavgpool2d", "adaptivemaxpool2d", "convtranspose", "convtranspose2d"):
                    if rank != 4:
                        errors.append(
                            f"Layer '{node.label}' ({node.type}) requires a 4D input tensor [Batch, Channels, Height, Width]. "
                            f"Received shape: {input_shape} (Rank {rank})."
                        )
                elif node_type in ("dense", "linear"):
                    if rank < 2:
                        errors.append(
                            f"Layer '{node.label}' ({node.type}) requires input tensor with rank >= 2. "
                            f"Received shape: {input_shape} (Rank {rank})."
                        )
                elif node_type in ("lstm", "gru", "rnn", "bidirectional", "multiheadattention", "mha"):
                    if rank != 3:
                        errors.append(
                            f"Layer '{node.label}' ({node.type}) requires a 3D input tensor [Batch, Seq_Len, Features]. "
                            f"Received shape: {input_shape} (Rank {rank})."
                        )
                elif node_type == "embedding":
                    if rank != 2:
                        errors.append(
                            f"Layer '{node.label}' ({node.type}) requires a 2D input tensor of token indexes [Batch, Seq_Len]. "
                            f"Received shape: {input_shape} (Rank {rank})."
                        )

            # 2. Multi-Head Attention Head Divisibility
            if node_type in ("multiheadattention", "mha"):
                embed_dim = config.get("embed_dim") or config.get("key_dim") or config.get("embedding_dim")
                if embed_dim is not None:
                    try:
                        embed_dim = int(embed_dim)
                    except ValueError:
                        embed_dim = None
                if embed_dim is None:
                    q_shape = input_shape[0] if isinstance(input_shape[0], list) else input_shape
                    if q_shape and len(q_shape) == 3:
                        embed_dim = q_shape[2]
                
                num_heads = int(config.get("num_heads", 8))
                if embed_dim is not None and num_heads > 0:
                    if embed_dim % num_heads != 0:
                        errors.append(
                            f"Multi-Head Attention layer '{node.label}' embedding dimension {embed_dim} "
                            f"is not perfectly divisible by num_heads {num_heads} (Remainder: {embed_dim % num_heads})."
                        )

            # 3. Reshape volume and dimension constraints
            if node_type == "reshape":
                target_shape = config.get("shape") or config.get("target_shape")
                if not target_shape:
                    errors.append(f"Reshape layer '{node.label}' requires a target 'shape' in its config.")
                elif not isinstance(target_shape, list):
                    errors.append(f"Reshape layer '{node.label}' target shape config must be a list.")
                else:
                    dynamic_count = sum(1 for d in target_shape if d in (-1, None, "None"))
                    if dynamic_count > 1:
                        errors.append(
                            f"Reshape layer '{node.label}' has illegal target shape {target_shape}. "
                            f"It contains {dynamic_count} dynamic dimensions (-1 or None), but at most one is allowed."
                        )
                    
                    if input_shape:
                        in_vol_concrete = True
                        in_vol = 1
                        for d in input_shape[1:]:
                            if d is None:
                                in_vol_concrete = False
                                break
                            in_vol *= int(d)
                            
                        if in_vol_concrete:
                            target_vol = 1
                            has_dynamic_target = False
                            for d in target_shape[1:]:
                                if d in (-1, None, "None"):
                                    has_dynamic_target = True
                                else:
                                    target_vol *= int(d)
                                    
                            if target_vol == 0:
                                errors.append(f"Reshape layer '{node.label}' has target dimensions of 0 which is illegal.")
                            elif not has_dynamic_target:
                                if in_vol != target_vol:
                                    errors.append(
                                        f"Reshape layer '{node.label}' volume mismatch. Input shape {input_shape} "
                                        f"(volume {in_vol}) cannot be reshaped to target shape {target_shape} (volume {target_vol})."
                                    )
                            else:
                                if in_vol % target_vol != 0:
                                    errors.append(
                                        f"Reshape layer '{node.label}' volume partition mismatch. Input shape {input_shape} "
                                        f"(volume {in_vol}) cannot be cleanly divided into target shape {target_shape} "
                                        f"(concrete volume {target_vol})."
                                    )

            # 4. Illegal Broadcasting for element-wise merges
            if node_type in ("add", "multiply"):
                if len(parent_shapes) >= 2:
                    from app.services.shape_inference_service import ShapeInferenceService
                    base_shape = parent_shapes[0]
                    for idx, other_shape in enumerate(parent_shapes[1:]):
                        try:
                            base_shape = ShapeInferenceService.broadcast_shapes(base_shape, other_shape)
                        except ValueError as e:
                            errors.append(
                                f"Operation layer '{node.label}' ({node.type}) has illegal broadcasting links. "
                                f"Parent 1 shape {parent_shapes[0]} vs Parent {idx+2} shape {other_shape} are incompatible. "
                                f"Detail: {e}"
                            )

        return errors

    @staticmethod
    def validate_framework_compatibility(nodes: List[Node], edges: List[Edge], target_framework: str) -> List[str]:
        """Validates the network graph against restrictions of the target execution frameworks (e.g. TensorFlow, ONNX).
        Returns a list of compatibility warnings or errors.
        """
        warnings: List[str] = []
        framework = target_framework.lower().strip()
        
        node_map = {node.id: node for node in nodes}
        incoming_edges: Dict[uuid.UUID, List[Edge]] = {nid: [] for nid in node_map}
        for edge in edges:
            if edge.to_node_id in incoming_edges:
                incoming_edges[edge.to_node_id].append(edge)

        for node in nodes:
            node_type = node.type.lower()
            config = node.config or {}
            
            input_shape = node.input_shape
            parents = incoming_edges[node.id]
            parent_shapes = []
            for p in parents:
                parent_node = node_map.get(p.from_node_id)
                if parent_node and parent_node.output_shape:
                    parent_shapes.append(parent_node.output_shape)
            if not input_shape and parent_shapes:
                input_shape = parent_shapes[0]

            # ONNX constraints
            if framework in ("onnx", "all"):
                if node_type in ("adaptivepool", "adaptiveavgpool", "adaptiveavgpool2d", "adaptivemaxpool2d"):
                    output_size = config.get("output_size")
                    if output_size:
                        try:
                            from app.services.shape_inference_service import ShapeInferenceService
                            parsed_size = ShapeInferenceService.parse_int_or_tuple(output_size)
                            if parsed_size != (1, 1):
                                warnings.append(
                                    f"[ONNX Compatibility Warning] Adaptive pooling layer '{node.label}' has output size {output_size}. "
                                    "ONNX export has limited support for adaptive pooling with target output resolutions other than [1, 1]."
                                )
                        except Exception:
                            pass

            # TensorFlow / Keras Functional API constraints
            if framework in ("tensorflow", "keras", "all"):
                if node_type == "embedding":
                    if input_shape and len(input_shape) != 2:
                        warnings.append(
                            f"[TensorFlow Compatibility Error] Embedding layer '{node.label}' expects exactly a 2D input [Batch, Seq_Len]. "
                            f"Received input shape: {input_shape} (Rank {len(input_shape)}). Keras Embedding does not natively support N-dimensional token tensors."
                        )
                if node_type in ("lstm", "gru", "rnn", "bidirectional"):
                    if input_shape and len(input_shape) != 3:
                        warnings.append(
                            f"[TensorFlow Compatibility Error] Recurrent layer '{node.label}' expects a 3D input. "
                            f"Received input shape: {input_shape}."
                        )

        return warnings
