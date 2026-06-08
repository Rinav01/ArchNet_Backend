from typing import List, Dict, Any
from app.models.node import Node
from app.models.edge import Edge
from app.services.automl_engine import AutoMLSuggestionEngine

class RecommendationEngine:
    @staticmethod
    def get_recommendations(nodes: List[Node], edges: List[Edge]) -> List[Dict[str, Any]]:
        """Analyzes network nodes and edges to produce highly customized,

        context-aware architectural recommendations referencing exact node labels,
        shapes, configurations, and parameter statistics.
        """
        # Get baseline recommendations from AutoMLSuggestionEngine
        base_recs = AutoMLSuggestionEngine.analyze_architecture_bottlenecks(nodes, edges)
        
        if not nodes:
            return base_recs

        custom_recs = []
        node_map = {str(node.id): node for node in nodes}
        
        # Build adjacency list
        adj_forward = {str(node.id): [] for node in nodes}
        adj_backward = {str(node.id): [] for node in nodes}
        for edge in edges:
            f_id, t_id = str(edge.from_node_id), str(edge.to_node_id)
            if f_id in adj_forward:
                adj_forward[f_id].append(t_id)
            if t_id in adj_backward:
                adj_backward[t_id].append(f_id)

        # Check for normalization layers presence
        has_normalization = False
        for node in nodes:
            ntype = getattr(node, "type", "").lower()
            if "batchnorm" in ntype or "layernorm" in ntype or "groupnorm" in ntype or "normalization" in ntype:
                has_normalization = True
                break

        # 1. "Add BatchNorm" Context-Specific Suggestion
        conv_nodes = [n for n in nodes if getattr(n, "type", "").lower() == "conv2d"]
        if conv_nodes and not has_normalization:
            # Recommends for the first Conv2D node found
            target_node = conv_nodes[0]
            out_shape_str = str(target_node.output_shape) if target_node.output_shape else "[None, channels, H, W]"
            custom_recs.append({
                "severity": "MEDIUM",
                "bottleneck": f"Add BatchNorm: Missing normalization layers after convolutions, starting at '{target_node.label}'.",
                "recommended_action": f"Add BatchNorm (such as a 'BatchNorm2D' layer) directly after convolutional layer '{target_node.label}' to normalize its output of shape {out_shape_str} before passing to downstream activations, preventing gradient vanishing."
            })

        # 2. "Reduce Dense Layer" Context-Specific Suggestion
        for node in nodes:
            if getattr(node, "type", "").lower() in ("dense", "linear"):
                config = node.config or {}
                units = int(config.get("units") or config.get("out_features", 0))
                if units > 512:
                    # Retrieve input features
                    in_features = 100
                    in_shape = node.input_shape
                    if in_shape:
                        flat_in = in_shape if not isinstance(in_shape[0], list) else in_shape[0]
                        if len(flat_in) > 0:
                            in_features = flat_in[-1] if flat_in[-1] is not None else 100
                    
                    param_count = units * in_features
                    reduced_units = 256
                    saved_params = (units - reduced_units) * in_features
                    savings_pct = round((saved_params / max(1, param_count)) * 100, 1)

                    custom_recs.append({
                        "severity": "HIGH",
                        "bottleneck": f"Reduce Dense Layer: Wide fully-connected projection '{node.label}' (units={units}) is a parameter hotspot.",
                        "recommended_action": f"Reduce Dense Layer '{node.label}' units from {units} to {reduced_units} to decrease parameters by {savings_pct}% (saving {saved_params:,} parameters), or insert a pooling layer to reduce the {in_features:,} input elements before projection."
                    })

        # 3. "Insert Residual Block" / Skip Connection Context-Specific Suggestion
        # Identify chains of convolutions
        if len(conv_nodes) >= 5:
            # Check if there is already a residual skip path (indicated by node with multiple inputs)
            has_residual = False
            for node_id, parents in adj_backward.items():
                if len(parents) > 1:
                    has_residual = True
                    break

            if not has_residual:
                src_node = conv_nodes[0]
                dest_node = conv_nodes[-1]
                custom_recs.append({
                    "severity": "MEDIUM",
                    "bottleneck": f"Insert Residual Block: Chain of {len(conv_nodes)} convolutions lacks skip/highway connections.",
                    "recommended_action": f"Insert Residual Block by adding a shortcut skip connection bypass from '{src_node.label}' to the output of '{dest_node.label}' (summing them using an 'Add' layer) to facilitate direct gradient propagation."
                })

        # Combine suggestions: remove default "fully validated" placeholder if custom ones exist
        final_recs = []
        for r in base_recs:
            if r.get("bottleneck") == "Architecture fully validated!" and custom_recs:
                continue
            final_recs.append(r)

        final_recs.extend(custom_recs)
        return final_recs
