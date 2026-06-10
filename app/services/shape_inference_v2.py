import math
import uuid
from typing import List, Dict, Any
from app.models.node import Node
from app.models.edge import Edge
from app.services.shape_inference_service import ShapeInferenceService

class ShapeInferenceServiceV2(ShapeInferenceService):
    @classmethod
    def infer_node_shape(cls, node: Node, input_shapes: List[List[Any]]) -> List[Any]:
        """Calculates the output shape for a single node based on its type, config, and incoming shapes list.
        Supports both vision [B, C, H, W] and sequence [B, T, D] shapes.
        """
        if input_shapes and not isinstance(input_shapes[0], list):
            input_shapes = [input_shapes]

        node_type = node.type.upper().strip()
        config = node.config or {}

        # If input_shapes is empty and it's not INPUT, default first_input
        first_input = input_shapes[0] if input_shapes else [None]
        batch_size = first_input[0] if first_input else None

        # ── Extended Sequence/Transformer Layers ─────────────────────────────
        if node_type == "EMBEDDING":
            # Input: [B, T]
            # Output: [B, T, D]
            seq_len = first_input[1] if len(first_input) > 1 else None
            embed_dim = int(config.get("embedding_dim") or config.get("output_dim", 128))
            return [batch_size, seq_len, embed_dim]

        elif node_type == "POSITIONAL_ENCODING":
            # Input: [B, T, D] ➔ Output: [B, T, D]
            return first_input

        elif node_type == "LAYER_NORM":
            # Input: [B, T, D] or [B, C, H, W] ➔ Output: same shape
            return first_input

        elif node_type in ("ATTENTION", "MULTI_HEAD_ATTENTION"):
            # Input: [B, T, D] (or list of query [B, T_q, D], key [B, T_k, D], value [B, T_v, D])
            # Output: [B, T_q, D_out]
            q_shape = first_input
            embed_dim = config.get("embed_dim") or config.get("key_dim") or config.get("embedding_dim")
            if embed_dim is not None:
                try:
                    embed_dim = int(embed_dim)
                except ValueError:
                    embed_dim = None

            if embed_dim is None:
                embed_dim = q_shape[2] if len(q_shape) > 2 else 128

            seq_len = q_shape[1] if len(q_shape) > 1 else None
            return [batch_size, seq_len, embed_dim]

        elif node_type == "RESIDUAL_ADD":
            # Element-wise addition shape broadcasting
            if len(input_shapes) < 2:
                return first_input
            out_shape = input_shapes[0]
            for shape in input_shapes[1:]:
                out_shape = cls.broadcast_shapes(out_shape, shape)
            return out_shape

        elif node_type in ("TRANSFORMER_BLOCK", "ENCODER_BLOCK", "DECODER_BLOCK"):
            # Input: [B, T, D] ➔ Output: [B, T, D]
            return first_input

        # ── Extended Recurrent Layers ────────────────────────────────────────
        elif node_type in ("LSTM", "GRU"):
            # Input: [B, T, D]
            # Output: [B, T, H] if return_sequences=True else [B, H]
            seq_len = first_input[1] if len(first_input) > 1 else None
            hidden_size = int(config.get("hidden_size") or config.get("units", 64))
            return_sequences = config.get("return_sequences", True)
            if isinstance(return_sequences, str):
                return_sequences = return_sequences.lower() == "true"
            else:
                return_sequences = bool(return_sequences)

            if return_sequences:
                return [batch_size, seq_len, hidden_size]
            else:
                return [batch_size, hidden_size]

        elif node_type == "BILSTM":
            # Input: [B, T, D]
            # Output: [B, T, 2H] if return_sequences=True else [B, 2H]
            seq_len = first_input[1] if len(first_input) > 1 else None
            hidden_size = int(config.get("hidden_size") or config.get("units", 64))
            return_sequences = config.get("return_sequences", True)
            if isinstance(return_sequences, str):
                return_sequences = return_sequences.lower() == "true"
            else:
                return_sequences = bool(return_sequences)

            if return_sequences:
                return [batch_size, seq_len, 2 * hidden_size]
            else:
                return [batch_size, 2 * hidden_size]

        # ── Extended Graph Neural Network Layers ──────────────────────────────
        elif node_type in ("GCN", "GRAPH_SAGE", "GAT"):
            # Input: [N, F] (node features) and [N, N] (adjacency)
            # Output: [N, F_out]
            num_nodes = first_input[0] if first_input else None
            out_features = int(config.get("out_features") or config.get("units") or config.get("hidden_size") or config.get("features", 64))
            return [num_nodes, out_features]

        # ── Fallback to legacy shape inference ────────────────────────────────
        # Normalize node for legacy parser
        legacy_node = Node(
            id=node.id,
            project_id=node.project_id,
            type=node.type,
            label=node.label,
            config=node.config,
            input_shape=node.input_shape,
            output_shape=node.output_shape
        )
        return ShapeInferenceService.infer_legacy_node_shape(legacy_node, input_shapes)

    @classmethod
    def run_shape_inference(cls, sorted_nodes: List[Node], edges: List[Edge]) -> None:
        """Traverses the sorted graph DAG and updates the input_shape and output_shape columns of each Node in place.
        Supports multi-input nodes and all new v2 sequence/GNN layers.
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

                # Run V2 shape inference
                out_shape = cls.infer_node_shape(node, parent_shapes)
                node.output_shape = out_shape
                node_output_shapes[node.id] = out_shape

                # Update outputs on the edge
                for edge in incoming:
                    edge.output_shape = out_shape
