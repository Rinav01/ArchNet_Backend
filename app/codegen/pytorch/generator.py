import os
import re
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict, Any
from app.models.node import Node
from app.models.project import Project

class PyTorchGenerator:
    @staticmethod
    def clean_variable_name(label: str) -> str:
        """Sanitize a label to make it a valid Python variable name."""
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', label.strip())
        # Replace multiple consecutive underscores
        cleaned = re.sub(r'_+', '_', cleaned).lower()
        if not cleaned:
            return "layer"
        if cleaned[0].isdigit():
            cleaned = "layer_" + cleaned
        return cleaned

    @staticmethod
    def get_activation_function(activation_str: Any) -> str | None:
        """Map config activation string to PyTorch/torch.nn.functional activations."""
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

    @classmethod
    def generate(cls, project: Project, sorted_nodes: List[Node]) -> str:
        """Generates standalone, runnable PyTorch code from a validated graph of nodes."""
        # Find directory of current file to find templates relative to it
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("model.py.jinja2")

        layers_init = []
        forward_steps = []
        
        # Track input shape details for dummy validation block
        input_shape_comment = "unknown"
        input_shape_test = "[1, 3, 224, 224]"
        test_input_dims = "1, 3, 224, 224"

        # Unique name tracker to avoid duplicate variable names
        name_counts: Dict[str, int] = {}

        for node in sorted_nodes:
            node_type = node.type.lower()
            if node_type == "input":
                # Save input shapes for dummy verification block
                shape = node.output_shape
                if shape:
                    input_shape_comment = str(shape)
                    # Replace None with 1 for test dims
                    test_dims = [dim if dim is not None else 1 for dim in shape]
                    input_shape_test = str(test_dims)
                    test_input_dims = ", ".join(map(str, test_dims))
                continue

            # Generate variable name
            base_var = cls.clean_variable_name(node.label or node.type)
            if base_var in name_counts:
                name_counts[base_var] += 1
                var_name = f"{base_var}_{name_counts[base_var]}"
            else:
                name_counts[base_var] = 1
                var_name = base_var

            config = node.config or {}
            
            # Map node type to PyTorch definition
            if node_type == "conv2d":
                in_channels = node.input_shape[1] if node.input_shape else 3
                filters = int(config.get("filters", 32))
                kernel_size = config.get("kernel_size", 3)
                stride = config.get("stride", 1)
                padding = config.get("padding", 0)

                # Format arguments nicely
                if isinstance(padding, str):
                    padding_str = f"'{padding}'"
                else:
                    padding_str = str(padding)

                init_str = f"nn.Conv2d(in_channels={in_channels}, out_channels={filters}, kernel_size={kernel_size}, stride={stride}, padding={padding_str})"
                layers_init.append({"name": var_name, "pytorch_init": init_str})
                
                # Check activation
                activation = cls.get_activation_function(config.get("activation"))
                forward_steps.append({
                    "name": var_name,
                    "shape": str(node.output_shape),
                    "activation": activation
                })

            elif node_type == "maxpool2d":
                pool_size = config.get("pool_size", 2)
                stride = config.get("stride", pool_size)
                padding = config.get("padding", 0)

                init_str = f"nn.MaxPool2d(kernel_size={pool_size}, stride={stride}, padding={padding})"
                layers_init.append({"name": var_name, "pytorch_init": init_str})
                
                forward_steps.append({
                    "name": var_name,
                    "shape": str(node.output_shape),
                    "activation": None
                })

            elif node_type == "flatten":
                init_str = "nn.Flatten(start_dim=1)"
                layers_init.append({"name": var_name, "pytorch_init": init_str})
                
                forward_steps.append({
                    "name": var_name,
                    "shape": str(node.output_shape),
                    "activation": None
                })

            elif node_type in ("dense", "linear"):
                in_features = node.input_shape[1] if node.input_shape else 10
                units = int(config.get("units") or config.get("out_features", 10))

                init_str = f"nn.Linear(in_features={in_features}, out_features={units})"
                layers_init.append({"name": var_name, "pytorch_init": init_str})
                
                # Check activation
                activation = cls.get_activation_function(config.get("activation"))
                forward_steps.append({
                    "name": var_name,
                    "shape": str(node.output_shape),
                    "activation": activation
                })
            else:
                # Fallback pass-through layer for any unknown block types
                init_str = "nn.Identity()"
                layers_init.append({"name": var_name, "pytorch_init": init_str})
                forward_steps.append({
                    "name": var_name,
                    "shape": str(node.output_shape),
                    "activation": None
                })

        # Render complete template
        rendered_code = template.render(
            project_name=project.name,
            framework=project.framework,
            layers_init=layers_init,
            forward_steps=forward_steps,
            input_shape_comment=input_shape_comment,
            input_shape_test=input_shape_test,
            test_input_dims=test_input_dims
        )
        
        return rendered_code
