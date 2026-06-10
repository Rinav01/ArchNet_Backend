from typing import List, Dict, Any
from app.models.node import Node
import math

class MemoryEstimator:
    @staticmethod
    def parse_int_or_tuple(val: Any) -> tuple[int, int]:
        if isinstance(val, int):
            return (val, val)
        if isinstance(val, (list, tuple)):
            if len(val) == 1:
                return (val[0], val[0])
            elif len(val) >= 2:
                return (val[0], val[1])
        return (3, 3) # default Conv fallback

    @staticmethod
    def estimate_node_metrics(node: Node, default_batch: int = 1) -> Dict[str, Any]:
        """Calculates parameters count, parameter memory, activation size, activation memory, and FLOPs for a single node."""
        node_type = node.type.lower()
        config = node.config or {}
        
        # Safe shape check helpers
        in_shape = node.input_shape
        out_shape = node.output_shape

        param_count = 0
        flops = 0

        # Resolving output size (replacing None with 1 for concrete calculations)
        out_size = 1
        if out_shape:
            # Flatten multi-nested shapes if they occur
            flat_out = out_shape if not isinstance(out_shape[0], list) else out_shape[0]
            for dim in flat_out:
                if dim is None:
                    out_size *= default_batch
                else:
                    out_size *= int(dim)

        if node_type == "conv2d" and in_shape and out_shape:
            # In shape: [Batch, Channels, Height, Width]
            in_channels = in_shape[1] if in_shape[1] is not None else 3
            filters = int(config.get("filters", 32))
            
            kernel_size = MemoryEstimator.parse_int_or_tuple(config.get("kernel_size", 3))
            kh, kw = kernel_size
            use_bias = bool(config.get("use_bias", True))
            
            # Param count: C_out * (C_in * Kh * Kw + Bias)
            param_count = filters * (in_channels * kh * kw + (1 if use_bias else 0))
            
            # Output height & width
            oh = out_shape[2] if out_shape[2] is not None else 1
            ow = out_shape[3] if out_shape[3] is not None else 1
            
            # FLOPs: 2 * C_in * C_out * Kh * Kw * H_out * W_out (MACs * 2)
            flops = 2 * (in_channels * kh * kw) * filters * oh * ow

        elif node_type in ("dense", "linear") and in_shape:
            # In shape: [Batch, ..., Features]
            in_features = in_shape[-1] if in_shape[-1] is not None else 10
            units = int(config.get("units") or config.get("out_features", 10))
            use_bias = bool(config.get("use_bias", True))
            
            # Param count: Units * (In_features + Bias)
            param_count = units * (in_features + (1 if use_bias else 0))
            
            # FLOPs: 2 * non-batch input size * Units
            non_batch_dims_product = 1
            for dim in in_shape[1:-1]:
                non_batch_dims_product *= (int(dim) if dim is not None else 1)
            flops = 2 * non_batch_dims_product * in_features * units

        out_size_for_mem = out_size

        if node_type in ("batchnorm", "batchnorm2d") and in_shape:
            in_channels = in_shape[1] if in_shape[1] is not None else 3
            # BatchNorm has 2 learnable parameters per channel
            param_count = 2 * in_channels
            # FLOPs: scaling operations (approx 2 FLOPs per activation element)
            flops = 2 * out_size

        elif node_type in ("layernorm", "layer_norm") and in_shape:
            embed_dim = in_shape[-1] if in_shape[-1] is not None else 128
            param_count = 2 * embed_dim
            # FLOPs: scaling operations (approx 2 FLOPs per activation element)
            flops = 2 * out_size

        elif node_type == "embedding":
            vocab_size = int(config.get("input_dim") or config.get("vocab_size", 1000))
            embed_dim = int(config.get("embedding_dim") or config.get("output_dim", 128))
            
            param_count = vocab_size * embed_dim
            flops = 0  # Table lookup has zero FLOPs

        elif node_type == "positional_encoding" and in_shape:
            embed_dim = config.get("embed_dim") or config.get("embedding_dim") or (in_shape[2] if len(in_shape) > 2 else 128)
            max_len = int(config.get("max_len", 1000))
            param_count = max_len * embed_dim
            seq_len = in_shape[1] if len(in_shape) > 1 and in_shape[1] is not None else 1
            flops = seq_len * embed_dim

        elif node_type in ("lstm", "gru", "rnn") and in_shape:
            # In shape: [Batch, Seq_Len, Features]
            in_features = in_shape[2] if len(in_shape) > 2 and in_shape[2] is not None else 64
            hidden_size = int(config.get("hidden_size") or config.get("units", 64))
            seq_len = in_shape[1] if len(in_shape) > 1 and in_shape[1] is not None else 1
            
            # Gate multiplier (LSTM = 4, GRU = 3, SimpleRNN = 1)
            gate_mult = 4 if node_type == "lstm" else (3 if node_type == "gru" else 1)
            
            # Param count: gate_mult * hidden_size * (in_features + hidden_size)
            param_count = gate_mult * hidden_size * (in_features + hidden_size)
            
            # FLOPs: 2 * gate_mult * seq_len * (in_features + hidden_size) * hidden_size
            flops = 2 * gate_mult * seq_len * (in_features + hidden_size) * hidden_size

        elif node_type in ("bidirectional", "bilstm") and in_shape:
            in_features = in_shape[2] if len(in_shape) > 2 and in_shape[2] is not None else 64
            hidden_size = int(config.get("hidden_size") or config.get("units", 64))
            seq_len = in_shape[1] if len(in_shape) > 1 and in_shape[1] is not None else 1
            
            # Bidirectional doubles parameters and FLOPs of underlying LSTM
            param_count = 2 * (4 * hidden_size * (in_features + hidden_size))
            flops = 2 * (8 * seq_len * (in_features + hidden_size) * hidden_size)

        elif node_type in ("maxpool2d", "avgpool", "avgpool2d") and in_shape and out_shape:
            in_channels = in_shape[1] if in_shape[1] is not None else 3
            pool_size = MemoryEstimator.parse_int_or_tuple(config.get("pool_size", 2))
            kh, kw = pool_size
            oh = out_shape[2] if out_shape[2] is not None else 1
            ow = out_shape[3] if out_shape[3] is not None else 1
            
            param_count = 0
            # Pooling FLOPs: C * Kh * Kw * H_out * W_out
            flops = in_channels * kh * kw * oh * ow

        elif node_type in ("convtranspose", "convtranspose2d") and in_shape and out_shape:
            in_channels = in_shape[1] if in_shape[1] is not None else 3
            filters = int(config.get("filters", 32))
            kernel_size = MemoryEstimator.parse_int_or_tuple(config.get("kernel_size", 3))
            kh, kw = kernel_size
            use_bias = bool(config.get("use_bias", True))
            
            param_count = filters * (in_channels * kh * kw + (1 if use_bias else 0))
            
            oh = out_shape[2] if out_shape[2] is not None else 1
            ow = out_shape[3] if out_shape[3] is not None else 1
            flops = 2 * (in_channels * kh * kw) * filters * oh * ow

        elif node_type in ("multiheadattention", "mha", "attention") and in_shape:
            # Query in_shape: [Batch, Seq_Len, Embed_Dim]
            q_shape = in_shape[0] if isinstance(in_shape[0], list) else in_shape
            embed_dim = config.get("embed_dim") or config.get("key_dim") or config.get("embedding_dim")
            if embed_dim is not None:
                try:
                    embed_dim = int(embed_dim)
                except ValueError:
                    embed_dim = None
            if embed_dim is None:
                embed_dim = q_shape[2] if len(q_shape) > 2 and q_shape[2] is not None else 128
                
            num_heads = int(config.get("num_heads", 8))
            seq_len = q_shape[1] if len(q_shape) > 1 and q_shape[1] is not None else 1
            
            # Learnable weights: 4 projection matrices (Q, K, V, Output)
            param_count = 4 * (embed_dim * embed_dim)
            
            # FLOPs: QK^2 * V self-attention FLOPs is roughly:
            # 2 * seq_len * seq_len * embed_dim (Q * K^T) + 2 * seq_len * seq_len * embed_dim (Softmax * V)
            flops = 4 * seq_len * seq_len * embed_dim
            # Attention Memory: Heads * T * T
            out_size_for_mem = out_size + default_batch * num_heads * seq_len * seq_len

        elif node_type in ("transformer_block", "encoder_block") and in_shape:
            q_shape = in_shape[0] if isinstance(in_shape[0], list) else in_shape
            embed_dim = config.get("embed_dim") or config.get("embedding_dim")
            if embed_dim is not None:
                try:
                    embed_dim = int(embed_dim)
                except ValueError:
                    embed_dim = None
            if embed_dim is None:
                embed_dim = q_shape[2] if len(q_shape) > 2 and q_shape[2] is not None else 128
                
            num_heads = int(config.get("num_heads", 8))
            seq_len = q_shape[1] if len(q_shape) > 1 and q_shape[1] is not None else 1
            
            # Param count: 12 * D^2
            param_count = 12 * (embed_dim * embed_dim)
            # FLOPs: 4 * T^2 * D + 24 * T * D^2
            flops = 4 * seq_len * seq_len * embed_dim + 24 * seq_len * embed_dim * embed_dim
            # Attention Memory: Heads * T * T
            out_size_for_mem = out_size + default_batch * num_heads * seq_len * seq_len

        elif node_type == "decoder_block" and in_shape:
            q_shape = in_shape[0] if isinstance(in_shape[0], list) and len(in_shape[0]) > 0 else in_shape
            embed_dim = config.get("embed_dim") or config.get("embedding_dim")
            if embed_dim is not None:
                try:
                    embed_dim = int(embed_dim)
                except ValueError:
                    embed_dim = None
            if embed_dim is None:
                embed_dim = q_shape[2] if len(q_shape) > 2 and q_shape[2] is not None else 128
                
            num_heads = int(config.get("num_heads", 8))
            seq_len = q_shape[1] if len(q_shape) > 1 and q_shape[1] is not None else 1
            
            # Param count: 16 * D^2
            param_count = 16 * (embed_dim * embed_dim)
            # FLOPs: 8 * T^2 * D + 32 * T * D^2
            flops = 8 * seq_len * seq_len * embed_dim + 32 * seq_len * embed_dim * embed_dim
            # Attention Memory: 2 * Heads * T * T
            out_size_for_mem = out_size + 2 * (default_batch * num_heads * seq_len * seq_len)

        elif node_type in ("gcn", "graph_sage", "gat") and in_shape:
            feat_shape = in_shape[0] if isinstance(in_shape[0], list) else in_shape
            num_nodes = feat_shape[0] if feat_shape and len(feat_shape) > 0 and feat_shape[0] is not None else 100
            in_features = feat_shape[1] if feat_shape and len(feat_shape) > 1 and feat_shape[1] is not None else 64
            out_features = int(config.get("out_features") or config.get("units") or config.get("hidden_size") or config.get("features", 64))
            
            if node_type == "gcn":
                # Param count: in_features * out_features
                param_count = in_features * out_features
                # FLOPs: 2 * Nodes * Output * (Input + Nodes)
                flops = 2 * num_nodes * out_features * (in_features + num_nodes)
            elif node_type == "graph_sage":
                # Param count: 2 * in_features * out_features
                param_count = 2 * in_features * out_features
                # FLOPs: 2 * Nodes * Input * (Nodes + 2 * Output)
                flops = 2 * num_nodes * in_features * (num_nodes + 2 * out_features)
            elif node_type == "gat":
                num_heads = int(config.get("num_heads", 1))
                # Param count: Heads * Input * Output + Heads * (2 * Output)
                param_count = num_heads * in_features * out_features + num_heads * (2 * out_features)
                # FLOPs: 2 * Nodes * Output * (Input + Nodes) * Heads
                flops = 2 * num_nodes * out_features * (in_features + num_nodes) * num_heads

        # General calculations
        # Memory assumes float32 (4 bytes per parameter / element)
        param_memory_mb = (param_count * 4) / (1024 * 1024)
        activation_memory_mb = (out_size_for_mem * 4) / (1024 * 1024)

        return {
            "parameter_count": param_count,
            "parameter_memory_mb": round(param_memory_mb, 4),
            "activation_size": out_size,
            "activation_memory_mb": round(activation_memory_mb, 4),
            "flops": float(flops)
        }

    @classmethod
    def estimate_project_metrics(cls, nodes: List[Node], default_batch: int = 1) -> Dict[str, Any]:
        """Aggregates memory and computational metrics across all nodes in the workspace DAG."""
        total_params = 0
        total_param_mem = 0.0
        total_act_mem = 0.0
        total_flops = 0.0

        for node in nodes:
            metrics = cls.estimate_node_metrics(node, default_batch)
            total_params += metrics["parameter_count"]
            total_param_mem += metrics["parameter_memory_mb"]
            total_act_mem += metrics["activation_memory_mb"]
            total_flops += metrics["flops"]

        return {
            "total_parameter_count": total_params,
            "total_parameter_memory_mb": round(total_param_mem, 4),
            "total_activation_memory_mb": round(total_act_mem, 4),
            "total_flops": total_flops,
            "estimated_gpu_memory_mb": round(total_param_mem + total_act_mem, 4)
        }
