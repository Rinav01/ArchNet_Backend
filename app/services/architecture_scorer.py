import math
from typing import List, Dict, Any
from app.models.node import Node
from app.models.edge import Edge
from app.services.memory_estimator import MemoryEstimator

class ArchitectureScorer:
    @staticmethod
    def score(nodes: List[Node], edges: List[Edge]) -> Dict[str, Any]:
        """Calculates a neural network score (0-100) and letter grade (A+ to F)

        based on six distinct architectural parameters:
        - Depth
        - Parameters
        - FLOPs
        - Memory
        - Gradient Stability
        - Regularization
        """
        if not nodes:
            return {
                "score": 0,
                "grade": "F",
                "breakdown": {
                    "depth": 0,
                    "parameters": 0,
                    "flops": 0,
                    "memory": 0,
                    "gradient_stability": 0,
                    "regularization": 0
                }
            }

        # 1. Depth Subscore
        depth = len(nodes)
        if depth < 3:
            depth_score = 30 * depth
        elif depth <= 30:
            depth_score = 100
        else:
            depth_score = max(50, 100 - (depth - 30) * 1)

        # Estimate project metrics using MemoryEstimator
        metrics = MemoryEstimator.estimate_project_metrics(nodes)
        total_params = metrics.get("total_parameter_count", 0)
        total_flops = metrics.get("total_flops", 0)
        estimated_gpu_mem = metrics.get("estimated_gpu_memory_mb", 0.0)

        # 2. Parameters Subscore
        if total_params == 0:
            param_score = 0
        elif total_params < 10000:
            param_score = 70
        elif total_params <= 50_000_000:
            param_score = 100
        else:
            param_score = max(30, 100 - int(math.log10(total_params / 50_000_000) * 20))

        # 3. FLOPs Subscore
        if total_flops == 0:
            flops_score = 100
        elif total_flops < 1e9: # < 1 GFLOP
            flops_score = 100
        else:
            flops_score = max(20, 100 - int(math.log10(total_flops / 1e9) * 15))

        # 4. Memory Subscore
        if estimated_gpu_mem < 500: # < 500 MB
            memory_score = 100
        elif estimated_gpu_mem < 8000: # < 8 GB
            memory_score = 90
        else:
            memory_score = max(10, 100 - int((estimated_gpu_mem - 8000) / 100))

        # 5. Gradient Stability Subscore
        # Heuristics:
        # - BatchNorm/LayerNorm/GroupNorm/Normalization presence: +30
        # - Residual skip connection (or if depth <= 6): +30
        # - Activations in conv/dense layers: +40
        has_norm = False
        has_residual = False
        
        # Check for norm layers
        for node in nodes:
            ntype = getattr(node, "type", getattr(node, "op_type", "")).lower()
            if "batchnorm" in ntype or "layernorm" in ntype or "groupnorm" in ntype or "normalization" in ntype:
                has_norm = True
                break
        
        # Check for residual connections
        # Adjacency maps to detect skip connections (if a node has >1 inputs, or a node has >1 outputs that merge later)
        adj_backward = {node.id: [] for node in nodes}
        for edge in edges:
            if edge.to_node_id in adj_backward:
                adj_backward[edge.to_node_id].append(edge.from_node_id)
        
        for node_id, parents in adj_backward.items():
            if len(parents) > 1:
                has_residual = True
                break

        norm_points = 30 if (has_norm or depth <= 4) else 0
        residual_points = 30 if (has_residual or depth <= 6) else 0

        # Check activations of operations
        op_nodes = [n for n in nodes if getattr(n, "type", getattr(n, "op_type", "")).lower() in ("conv2d", "dense", "linear")]
        if not op_nodes:
            activation_points = 40
        else:
            nodes_with_activation = 0
            for node in op_nodes:
                cfg = getattr(node, "config", getattr(node, "params", {}))
                act = cfg.get("activation")
                if act and act.lower() not in ("none", "linear"):
                    nodes_with_activation += 1
            activation_points = int(40 * (nodes_with_activation / len(op_nodes)))

        grad_stability_score = norm_points + residual_points + activation_points
        grad_stability_score = min(100, max(10, grad_stability_score))

        # 6. Regularization Subscore
        # Heuristics:
        # - Dropout presence: +60
        # - BatchNorm (acts as weak regularization): +20
        # - L2 / weight decay or other: +20
        # - If depth <= 3, default to 100
        if depth <= 3:
            reg_score = 100
        else:
            has_dropout = any(getattr(node, "type", getattr(node, "op_type", "")).lower() == "dropout" for node in nodes)
            # Check if any dense layers are wide (> 128 units). If none, we don't strictly need dropout
            has_wide_dense = any(
                getattr(node, "type", getattr(node, "op_type", "")).lower() in ("dense", "linear") and 
                int(getattr(node, "config", getattr(node, "params", {})).get("units") or getattr(node, "config", getattr(node, "params", {})).get("out_features", 0)) > 128
                for node in nodes
            )
            dropout_points = 60 if (has_dropout or not has_wide_dense) else 0
            norm_points_reg = 20 if has_norm else 0
            
            # Check weight decay / L2 / regularization config in any layer config or training setting
            has_l2 = any(
                "weight_decay" in str(getattr(node, "config", getattr(node, "params", {}))).lower() or 
                "regularizer" in str(getattr(node, "config", getattr(node, "params", {}))).lower()
                for node in nodes
            )
            l2_points = 20 if has_l2 else 0
            
            reg_score = dropout_points + norm_points_reg + l2_points
            reg_score = min(100, max(30, reg_score))

        # Calculate final weighted score
        final_score = int(
            depth_score * 0.15 +
            param_score * 0.15 +
            flops_score * 0.15 +
            memory_score * 0.15 +
            grad_stability_score * 0.25 +
            reg_score * 0.15
        )
        final_score = min(100, max(0, final_score))

        # Grade mapping
        if final_score >= 95:
            grade = "A+"
        elif final_score >= 90:
            grade = "A"
        elif final_score >= 85:
            grade = "A-"
        elif final_score >= 80:
            grade = "B+"
        elif final_score >= 75:
            grade = "B"
        elif final_score >= 70:
            grade = "B-"
        elif final_score >= 65:
            grade = "C+"
        elif final_score >= 60:
            grade = "C"
        elif final_score >= 55:
            grade = "C-"
        elif final_score >= 50:
            grade = "D"
        else:
            grade = "F"

        return {
            "score": final_score,
            "grade": grade,
            "breakdown": {
                "depth": depth_score,
                "parameters": param_score,
                "flops": flops_score,
                "memory": memory_score,
                "gradient_stability": grad_stability_score,
                "regularization": reg_score
            }
        }
