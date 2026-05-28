from typing import List, Dict, Any
from app.models.node import Node
from app.models.edge import Edge

class AutoMLSuggestionEngine:
    @staticmethod
    def analyze_architecture_bottlenecks(nodes: List[Node], edges: List[Edge]) -> List[Dict[str, Any]]:
        """Analyzes a neural network DAG structure to detect performance bottlenecks,
        vanishing gradient risks, missing activations/pooling, and suggests architectural optimizations.
        """
        recommendations = []
        if not nodes:
            return recommendations

        # Map nodes by ID for fast lookup
        node_map = {node.id: node for node in nodes}
        
        # Build adjacency maps representing DAG structure
        adj_forward = {node.id: [] for node in nodes}
        adj_backward = {node.id: [] for node in nodes}
        for edge in edges:
            if edge.from_node_id in adj_forward:
                adj_forward[edge.from_node_id].append(edge.to_node_id)
            if edge.to_node_id in adj_backward:
                adj_backward[edge.to_node_id].append(edge.from_node_id)

        # 1. Detect Dense-after-Conv Parameter Explosion
        for node in nodes:
            node_type = node.type.lower()
            if node_type in ("dense", "linear"):
                config = node.config or {}
                units = int(config.get("units") or config.get("out_features", 10))
                
                # Trace backwards to check if there is a Conv2D but no MaxPool/GlobalAvgPool
                parents = adj_backward.get(node.id, [])
                for parent_id in parents:
                    parent = node_map.get(parent_id)
                    if parent and parent.type.lower() in ("conv2d", "convtranspose2d"):
                        # We have a direct link from Conv2D to Dense! This is a massive parameter bottleneck!
                        in_shape = node.input_shape
                        if in_shape and len(in_shape) == 4:
                            # e.g. [Batch, 64, 224, 224] going directly to Dense
                            c_in, h_in, w_in = in_shape[1], in_shape[2], in_shape[3]
                            if h_in and w_in and h_in * w_in > 49: # spatial size larger than 7x7
                                flat_size = c_in * h_in * w_in
                                raw_params = flat_size * units
                                
                                # Estimate params with GlobalAveragePooling (spatial HxW reduced to 1x1)
                                gap_params = c_in * units
                                savings = raw_params - gap_params
                                savings_pct = round((savings / raw_params) * 100, 2)
                                
                                recommendations.append({
                                    "severity": "HIGH",
                                    "bottleneck": f"Parameter Explosion detected at '{node.label}'! Connecting 4D Convolution output {in_shape} directly to a Dense layer creates a massive flat size of {flat_size} elements, yielding {raw_params:,} learnable parameters.",
                                    "recommended_action": f"Insert a 'GlobalAveragePooling2D' or 'MaxPool2D' layer between '{parent.label}' and '{node.label}' to downsample spatial dimensions. This will reduce parameters by up to {savings_pct}% (saving {savings:,} parameters) and prevent out-of-memory errors."
                                })

        # 2. Detect Vanishing Gradient Risk (Deep networks without BatchNorm or Residual links)
        # We calculate max convolutional chain depth
        conv_depths = {node.id: 0 for node in nodes}
        has_batchnorm = False
        
        # Simple topological layer scanner to compute conv chain sizes
        for node in nodes:
            node_type = node.type.lower()
            if "batchnorm" in node_type:
                has_batchnorm = True
            
            if node_type in ("conv2d", "dense", "linear"):
                parents = adj_backward.get(node.id, [])
                max_parent_depth = 0
                for parent_id in parents:
                    max_parent_depth = max(max_parent_depth, conv_depths.get(parent_id, 0))
                conv_depths[node.id] = max_parent_depth + 1

        max_depth = max(conv_depths.values()) if conv_depths else 0
        if max_depth > 6 and not has_batchnorm:
            recommendations.append({
                "severity": "MEDIUM",
                "bottleneck": f"Vanishing Gradient Risk: Network contains a deep operational layer chain of {max_depth} layers without any Batch Normalization layers.",
                "recommended_action": "Inject 'BatchNorm2D' or 'BatchNorm' layers after convolutions or dense projections. This standardizes activation scales, prevents vanishing/exploding gradients, and speeds up training convergence."
            })

        # 3. Detect Missing Regularization / Overfitting Risk (Wide Linear blocks)
        for node in nodes:
            node_type = node.type.lower()
            if node_type in ("dense", "linear"):
                config = node.config or {}
                units = int(config.get("units") or config.get("out_features", 10))
                
                if units > 512:
                    # Check if there is a Dropout layer directly connected forwards
                    children = adj_forward.get(node.id, [])
                    has_dropout = False
                    for child_id in children:
                        child = node_map.get(child_id)
                        if child and child.type.lower() == "dropout":
                            has_dropout = True
                            
                    if not has_dropout:
                        recommendations.append({
                            "severity": "MEDIUM",
                            "bottleneck": f"High Overfitting Risk at '{node.label}'! Very wide projection layer detected (units = {units}) without any regularizing Dropout nodes.",
                            "recommended_action": f"Connect a 'Dropout' layer (rate=0.3 to 0.5) directly after '{node.label}' to randomly mute activations during training, enforcing robust generalized feature representation."
                        })

        # 4. Detect Missing Activation Function
        for node in nodes:
            node_type = node.type.lower()
            if node_type in ("conv2d", "dense", "linear"):
                config = node.config or {}
                activation = config.get("activation")
                
                if not activation or activation.lower() in ("linear", "none"):
                    # Check if the child also lacks activation
                    children = adj_forward.get(node.id, [])
                    for child_id in children:
                        child = node_map.get(child_id)
                        if child and child.type.lower() in ("conv2d", "dense", "linear"):
                            child_activation = (child.config or {}).get("activation")
                            if not child_activation or child_activation.lower() in ("linear", "none"):
                                recommendations.append({
                                    "severity": "LOW",
                                    "bottleneck": f"Consecutive Linear Operations: Both '{node.label}' and '{child.label}' represent purely linear transformations.",
                                    "recommended_action": f"Specify a non-linear activation function (such as 'ReLU' or 'SiLU') in the settings of '{node.label}' or insert an activation layer. Multi-layer models collapse to a single linear layer if non-linear activations are omitted."
                                })
                                break

        # If architecture is fully optimized
        if not recommendations:
            recommendations.append({
                "severity": "NONE",
                "bottleneck": "Architecture fully validated!",
                "recommended_action": "Neural network DAG matches all modern structural design guidelines. Ready for dynamic benchmarking and high-performance cloud training!"
            })

        return recommendations
