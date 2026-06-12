import uuid
from sqlalchemy.orm import Session
from app.models.project import Project
from app.models.node import Node
from app.models.edge import Edge
from app.services.memory_estimator import MemoryEstimator
from app.services.validation_service import ValidationService
from app.services.shape_inference_service import ShapeInferenceService

class ExplainabilityAgent:
    @staticmethod
    def generate_explanation(db: Session, project_id: uuid.UUID) -> dict:
        """
        Generates structured Markdown reports outlining tensor shape transitions, attention mechanics, 
        VRAM overhead distributions, parameter formulas, and code-generator compiler logic.
        """
        # Fetch the project entity
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError("Project not found.")

        # Load node and edge graphs
        nodes = db.query(Node).filter(Node.project_id == project_id).all()
        edges = db.query(Edge).filter(Edge.project_id == project_id).all()

        # Run topological sorts and shape inference queries
        sorted_nodes = []
        try:
            sorted_nodes = ValidationService.validate_graph(nodes, edges)
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
            db.commit()
            # Reload nodes to get computed shape properties
            nodes = db.query(Node).filter(Node.project_id == project_id).all()
            sorted_nodes = ValidationService.validate_graph(nodes, edges)
        except Exception:
            sorted_nodes = nodes

        # 1. Shape Propagation Analysis
        shape_explanation = ["### Shape Propagation Analysis\n"]
        shape_explanation.append("Tensor dimensions propagate through the graph sequentially according to layer operations:")
        shape_explanation.append("| Layer Label | Layer Type | Input Shape | Output Shape | Transition Explanation |")
        shape_explanation.append("| :--- | :--- | :--- | :--- | :--- |")
        
        for node in sorted_nodes:
            in_s = str(node.input_shape) if node.input_shape else "None"
            out_s = str(node.output_shape) if node.output_shape else "None"
            node_type_lower = node.type.lower()
            explanation = "Maintains input dimension shape."
            
            if node_type_lower in ("dense", "linear"):
                units = node.config.get("units") or node.config.get("out_features", "N/A")
                explanation = f"Projects the feature dimension to {units} output features."
            elif node_type_lower == "conv2d":
                filters = node.config.get("filters", 32)
                explanation = f"Applies 2D spatial cross-correlation projecting to {filters} output channels."
            elif node_type_lower == "flatten":
                explanation = "Flattens spatial dimensions into a single feature vector."
            elif node_type_lower in ("maxpool2d", "avgpool2d", "avgpool"):
                pool_size = node.config.get("pool_size", "2")
                explanation = f"Reduces spatial height and width by pooling with kernel size {pool_size}."
            elif node_type_lower == "embedding":
                explanation = "Maps sparse token indices to dense continuous embeddings."
            elif node_type_lower in ("lstm", "gru", "rnn"):
                units = node.config.get("hidden_size") or node.config.get("units", "N/A")
                explanation = f"Recurrent sequential propagation mapping sequence to hidden dimension size {units}."
                
            shape_explanation.append(f"| **{node.label}** | {node.type} | `{in_s}` | `{out_s}` | {explanation} |")
        
        shape_propagation_md = "\n".join(shape_explanation)

        # 2. Attention Scaling Analysis
        attention_nodes = [n for n in sorted_nodes if n.type.lower() in ("multiheadattention", "mha", "attention", "transformer_block", "encoder_block", "decoder_block")]
        if attention_nodes:
            attention_explanation = ["### Attention Scaling Analysis\n"]
            for node in attention_nodes:
                config = node.config or {}
                num_heads = config.get("num_heads", 8)
                embed_dim = config.get("embed_dim") or config.get("embedding_dim") or 128
                attention_explanation.append(f"#### Layer: {node.label} ({node.type})")
                attention_explanation.append(f"- **Projections**: Input features map to Query, Key, and Value vectors of size `{embed_dim}` across `{num_heads}` heads.")
                attention_explanation.append(f"- **Scaling Factor**: Softmax scaling utilizes $1/\\sqrt{{d_k}} = 1/\\sqrt{{{embed_dim} / {num_heads}}} = {round(1.0 / ((embed_dim / num_heads) ** 0.5), 4):.4f}$ to prevent gradient saturation at high dimensions.")
                attention_explanation.append(f"- **Complexity**: The self-attention matrix requires computing pairwise scores mapping to a spatial complexity of $O(B \\cdot H \\cdot T^2)$ (currently $O({num_heads} \\cdot T^2)$) and computational complexity of $O(B \\cdot T^2 \\cdot D)$ where $T$ is sequence length.")
        else:
            attention_explanation = [
                "### Attention Scaling Analysis\n",
                "No Attention or Transformer-based layers detected in this project graph.",
                "Self-attention scales activation dot products using the formula:",
                "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$",
                "Where $d_k$ represents the Query/Key dimension per head. This scaling factor division prevents the dot products from growing excessively in magnitude, which would push the softmax function into regions with extremely small gradients."
            ]
        attention_scaling_md = "\n".join(attention_explanation)

        # 3. VRAM Memory Allocation Breakdown
        project_metrics = MemoryEstimator.estimate_project_metrics(sorted_nodes)
        total_param_mem = project_metrics["total_parameter_memory_mb"]
        total_act_mem = project_metrics["total_activation_memory_mb"]
        total_gpu_mem = project_metrics["estimated_gpu_memory_mb"]

        vram_explanation = [
            "### VRAM Memory Allocation Breakdown\n",
            f"Estimated peak GPU VRAM required: **{total_gpu_mem:.4f} MB**",
            f"- **Weights & Parameters Memory**: **{total_param_mem:.4f} MB** (Static allocation for layer coefficients).",
            f"- **Activation Memory**: **{total_act_mem:.4f} MB** (Dynamic storage for intermediate tensor forward outputs at batch size 1).",
            "- **Training Overhead Estimate**: **{:.4f} MB** (Additional space for gradients and Adam optimizer states, typically estimated at $2 \\times \\text{{Parameters Memory}} + \\text{{Activation Memory}}$).".format(2 * total_param_mem + total_act_mem)
        ]
        vram_usage_md = "\n".join(vram_explanation)

        # 4. Parameter Calculations Details
        param_explanation = ["### Parameter Calculation Details\n"]
        param_explanation.append("| Layer Label | Layer Type | Config Properties | Parameter Calculation Formula | Total Parameters |")
        param_explanation.append("| :--- | :--- | :--- | :--- | :--- |")
        
        for node in sorted_nodes:
            metrics = MemoryEstimator.estimate_node_metrics(node)
            param_count = metrics["parameter_count"]
            node_type_lower = node.type.lower()
            config = node.config or {}
            
            formula = "0"
            properties = []
            
            if node_type_lower == "conv2d" and node.input_shape:
                in_c = node.input_shape[1] if node.input_shape[1] is not None else 3
                out_c = config.get("filters", 32)
                kh, kw = MemoryEstimator.parse_int_or_tuple(config.get("kernel_size", 3))
                formula = f"filters * (in_channels * kh * kw + bias) = {out_c} * ({in_c} * {kh} * {kw} + 1)"
                properties = [f"filters={out_c}", f"kernel={kh}x{kw}"]
            elif node_type_lower in ("dense", "linear") and node.input_shape:
                in_f = node.input_shape[-1] if node.input_shape[-1] is not None else 10
                units = config.get("units") or config.get("out_features", 10)
                formula = f"units * (in_features + bias) = {units} * ({in_f} + 1)"
                properties = [f"units={units}"]
            elif node_type_lower == "embedding":
                vocab = config.get("input_dim") or config.get("vocab_size", 1000)
                dim = config.get("embedding_dim") or config.get("output_dim", 128)
                formula = f"vocab_size * embedding_dim = {vocab} * {dim}"
                properties = [f"vocab={vocab}", f"dim={dim}"]
            elif node_type_lower in ("lstm", "gru", "rnn") and node.input_shape:
                in_f = node.input_shape[2] if len(node.input_shape) > 2 and node.input_shape[2] is not None else 64
                hidden = config.get("hidden_size") or config.get("units", 64)
                gate_mult = 4 if node_type_lower == "lstm" else (3 if node_type_lower == "gru" else 1)
                formula = f"gate_mult * hidden * (in_features + hidden) = {gate_mult} * {hidden} * ({in_f} + {hidden})"
                properties = [f"hidden={hidden}", f"gates={gate_mult}"]
            else:
                formula = "No learnable parameters parameters."
                properties = [f"type={node.type}"]

            properties_str = ", ".join(properties)
            param_explanation.append(f"| **{node.label}** | {node.type} | {properties_str} | `{formula}` | {param_count:,} |")

        parameter_counts_md = "\n".join(param_explanation)

        # 5. Compiler Decisions Report
        framework = project.framework or "PyTorch"
        compiler_explanation = [
            "### Compiler Decision Report\n",
            f"Target framework compilation: **{framework}**",
            "- **Topological Sort**: Kahn's algorithm is utilized to resolve structural dependencies, ensuring preceding layers compile before dependent nodes are initialized.",
            f"- **Graph Partitioning**: Visual node connections map to a unified sub-module class `{framework}` format (e.g. `class MLBuilderModel` in PyTorch).",
            "- **Fusing & Simplifications**: Consecutive layer combinations (like Conv2D and ReLU activations) are statically consolidated into inline operations where possible to optimize runtime execution graphs."
        ]
        compiler_decisions_md = "\n".join(compiler_explanation)

        return {
            "shape_propagation": shape_propagation_md,
            "attention_scaling": attention_scaling_md,
            "vram_usage": vram_usage_md,
            "parameter_counts": parameter_counts_md,
            "compiler_decisions": compiler_decisions_md
        }
