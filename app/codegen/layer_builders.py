import re
from typing import Dict, Any, List, Tuple

def clean_variable_name(label: str) -> str:
    """Sanitize a label to make it a valid Python variable name."""
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', label.strip())
    cleaned = re.sub(r'_+', '_', cleaned).lower()
    if not cleaned:
        return "layer"
    if cleaned[0].isdigit():
        cleaned = "layer_" + cleaned
    return cleaned

def get_activation_function(activation_str: Any) -> str | None:
    if not activation_str or not isinstance(activation_str, str):
        return None
    act = activation_str.lower().strip()
    if act == "relu":
        return "F.relu"
    elif act == "sigmoid":
        return "torch.sigmoid"
    elif act == "tanh":
        return "torch.tanh"
    elif act in ("softmax", "log_softmax"):
        return f"lambda t: F.{act}(t, dim=1)"
    return None

def build_layer(op_type: str, label: str, params: Dict[str, Any], input_shape: Any, inputs: List[str], node_vars: Dict[str, str]) -> Tuple[str | None, str, bool, str | None]:
    """
    Returns:
        init_str: The initialization code (e.g. "nn.Conv2d(...)") or None for tensor operations.
        forward_str: The line of code for the forward pass (e.g. "x = self.conv2d(x)").
        is_tensor_op: True if it is a pure tensor operation and requires no nn.Module instantiation.
        activation: Functional activation string (e.g. "F.relu") if configured in params.
    """
    op_lower = op_type.lower().strip()
    var_name = node_vars.get(label) or clean_variable_name(label)
    
    # Resolve inputs
    if not inputs:
        input_arg = "x"
    elif len(inputs) == 1:
        input_arg = node_vars.get(inputs[0], "x")
    else:
        input_arg = ", ".join([node_vars.get(inp, "x") for inp in inputs])

    is_tensor_op = False
    init_str = None
    forward_str = None
    activation = None

    # Handle standard layers
    if op_lower == "input":
        is_tensor_op = True
        forward_str = f"{var_name} = x"
        
    elif op_lower in ("conv2d", "conv_2d"):
        in_channels = input_shape[1] if input_shape and len(input_shape) > 1 else 3
        filters = int(params.get("filters") or params.get("out_channels", 32))
        kernel_size = params.get("kernel_size", 3)
        stride = params.get("stride", 1)
        padding = params.get("padding", 0)
        padding_str = f"'{padding}'" if isinstance(padding, str) else str(padding)
        init_str = f"nn.Conv2d(in_channels={in_channels}, out_channels={filters}, kernel_size={kernel_size}, stride={stride}, padding={padding_str})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"
        activation = get_activation_function(params.get("activation"))
        
    elif op_lower in ("dense", "linear"):
        in_features = input_shape[-1] if input_shape else 10
        units = int(params.get("units") or params.get("out_features", 10))
        init_str = f"nn.Linear(in_features={in_features}, out_features={units})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"
        activation = get_activation_function(params.get("activation"))
        
    elif op_lower == "relu":
        init_str = "nn.ReLU()"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"
        
    elif op_lower == "sigmoid":
        init_str = "nn.Sigmoid()"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"
        
    elif op_lower == "tanh":
        init_str = "nn.Tanh()"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"

    elif op_lower in ("softmax", "log_softmax"):
        init_str = "nn.Softmax(dim=1)" if op_lower == "softmax" else "nn.LogSoftmax(dim=1)"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"

    elif op_lower in ("maxpool", "maxpool2d", "max_pool", "max_pool2d"):
        pool_size = params.get("pool_size", 2)
        stride = params.get("stride", pool_size)
        padding = params.get("padding", 0)
        init_str = f"nn.MaxPool2d(kernel_size={pool_size}, stride={stride}, padding={padding})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"
        
    elif op_lower in ("avgpool", "avgpool2d", "avg_pool", "avg_pool2d"):
        pool_size = params.get("pool_size", 2)
        stride = params.get("stride", pool_size)
        padding = params.get("padding", 0)
        init_str = f"nn.AvgPool2d(kernel_size={pool_size}, stride={stride}, padding={padding})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"

    elif op_lower in ("adaptiveavgpool", "adaptiveavgpool2d", "adaptive_avg_pool", "adaptive_avg_pool2d"):
        output_size = params.get("output_size", 1)
        init_str = f"nn.AdaptiveAvgPool2d(output_size={output_size})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"

    elif op_lower in ("adaptivemaxpool", "adaptivemaxpool2d", "adaptive_max_pool", "adaptive_max_pool2d"):
        output_size = params.get("output_size", 1)
        init_str = f"nn.AdaptiveMaxPool2d(output_size={output_size})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"

    elif op_lower in ("convtranspose", "convtranspose2d", "conv_transpose", "conv_transpose2d"):
        in_channels = input_shape[1] if input_shape and len(input_shape) > 1 else 3
        filters = int(params.get("filters", 32))
        kernel_size = params.get("kernel_size", 3)
        stride = params.get("stride", 1)
        padding = params.get("padding", 0)
        output_padding = params.get("output_padding", 0)
        init_str = f"nn.ConvTranspose2d(in_channels={in_channels}, out_channels={filters}, kernel_size={kernel_size}, stride={stride}, padding={padding}, output_padding={output_padding})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"

    elif op_lower == "flatten":
        start_dim = int(params.get("start_dim", 1))
        init_str = f"nn.Flatten(start_dim={start_dim})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"
        
    elif op_lower in ("batchnorm", "batchnorm2d", "batch_norm", "batch_norm2d"):
        in_features = input_shape[1] if input_shape and len(input_shape) > 1 else 3
        init_str = f"nn.BatchNorm2d(num_features={in_features})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"
        
    elif op_lower in ("layernorm", "layer_norm"):
        normalized_shape = params.get("normalized_shape")
        if not normalized_shape and input_shape:
            normalized_shape = input_shape[1:]
        init_str = f"nn.LayerNorm(normalized_shape={normalized_shape})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"

    elif op_lower == "dropout":
        p = float(params.get("p", 0.5))
        init_str = f"nn.Dropout(p={p})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"
        
    elif op_lower in ("lstm", "gru", "rnn"):
        in_features = input_shape[2] if input_shape and len(input_shape) > 2 else 64
        hidden_size = int(params.get("hidden_size") or params.get("units", 64))
        num_layers = int(params.get("num_layers", 1))
        bidirectional = bool(params.get("bidirectional", False))
        
        rnn_class = "LSTM" if op_lower == "lstm" else ("GRU" if op_lower == "gru" else "RNN")
        init_str = f"nn.{rnn_class}(input_size={in_features}, hidden_size={hidden_size}, num_layers={num_layers}, batch_first=True, bidirectional={bidirectional})"
        
        return_sequences = bool(params.get("return_sequences", False))
        if return_sequences:
            forward_str = f"{var_name}, _ = self.{var_name}({input_arg})"
        else:
            if op_lower == "lstm":
                forward_str = f"_, (hn, _) = self.{var_name}({input_arg})\n        if self.{var_name}.bidirectional:\n            {var_name} = torch.cat((hn[-2], hn[-1]), dim=-1)\n        else:\n            {var_name} = hn[-1]"
            else:
                forward_str = f"_, hn = self.{var_name}({input_arg})\n        if self.{var_name}.bidirectional:\n            {var_name} = torch.cat((hn[-2], hn[-1]), dim=-1)\n        else:\n            {var_name} = hn[-1]"

    elif op_lower == "embedding":
        vocab_size = int(params.get("input_dim") or params.get("vocab_size", 1000))
        embed_dim = int(params.get("embedding_dim") or params.get("output_dim", 128))
        init_str = f"nn.Embedding(num_embeddings={vocab_size}, embedding_dim={embed_dim})"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"

    elif op_lower in ("multiheadattention", "mha"):
        embed_dim = params.get("embed_dim") or params.get("key_dim") or params.get("embedding_dim")
        if embed_dim is not None:
            embed_dim = int(embed_dim)
        else:
            q_shape = input_shape[0] if input_shape and isinstance(input_shape[0], list) else input_shape
            embed_dim = q_shape[2] if q_shape and len(q_shape) > 2 else 128
        num_heads = int(params.get("num_heads", 8))
        init_str = f"nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={num_heads}, batch_first=True)"
        
        parents = [node_vars.get(pid, "x") for pid in inputs]
        if len(parents) == 3:
            forward_str = f"{var_name}, _ = self.{var_name}({parents[0]}, {parents[1]}, {parents[2]})"
        elif len(parents) == 2:
            forward_str = f"{var_name}, _ = self.{var_name}({parents[0]}, {parents[1]}, {parents[1]})"
        else:
            forward_str = f"{var_name}, _ = self.{var_name}({input_arg}, {input_arg}, {input_arg})"

    elif op_lower == "add":
        is_tensor_op = True
        parents = [node_vars.get(pid, "x") for pid in inputs]
        sum_arg = " + ".join(parents)
        forward_str = f"{var_name} = {sum_arg}"

    elif op_lower == "subtract":
        is_tensor_op = True
        parents = [node_vars.get(pid, "x") for pid in inputs]
        if len(parents) >= 2:
            sub_arg = " - ".join(parents)
            forward_str = f"{var_name} = {sub_arg}"
        else:
            forward_str = f"{var_name} = {input_arg}"

    elif op_lower == "multiply":
        is_tensor_op = True
        parents = [node_vars.get(pid, "x") for pid in inputs]
        mul_arg = " * ".join(parents)
        forward_str = f"{var_name} = {mul_arg}"

    elif op_lower == "maximum":
        is_tensor_op = True
        parents = [node_vars.get(pid, "x") for pid in inputs]
        if len(parents) >= 2:
            cur = parents[0]
            for other in parents[1:]:
                cur = f"torch.maximum({cur}, {other})"
            forward_str = f"{var_name} = {cur}"
        else:
            forward_str = f"{var_name} = {input_arg}"

    elif op_lower == "minimum":
        is_tensor_op = True
        parents = [node_vars.get(pid, "x") for pid in inputs]
        if len(parents) >= 2:
            cur = parents[0]
            for other in parents[1:]:
                cur = f"torch.minimum({cur}, {other})"
            forward_str = f"{var_name} = {cur}"
        else:
            forward_str = f"{var_name} = {input_arg}"

    elif op_lower == "concatenate":
        is_tensor_op = True
        axis = int(params.get("axis", 1))
        parents = [node_vars.get(pid, "x") for pid in inputs]
        forward_str = f"{var_name} = torch.cat(({', '.join(parents)}), dim={axis})"

    elif op_lower == "reshape":
        is_tensor_op = True
        target_shape = params.get("shape") or params.get("target_shape", [-1])
        cleaned_shape = [dim if dim is not None else -1 for dim in target_shape]
        forward_str = f"{var_name} = torch.reshape({input_arg}, {cleaned_shape})"

    elif op_lower in ("permute", "transpose"):
        is_tensor_op = True
        axes = params.get("axes") or params.get("dims", [0, 2, 1, 3])
        forward_str = f"{var_name} = {input_arg}.permute({axes})"

    else:
        init_str = "nn.Identity()"
        forward_str = f"{var_name} = self.{var_name}({input_arg})"

    return init_str, forward_str, is_tensor_op, activation
