import uuid
from typing import Dict, Any, List

class ArchitectureTemplateGenerator:
    @staticmethod
    def generate(prompt: str) -> Dict[str, Any]:
        """Maps natural language prompts (e.g., 'Image classifier for CIFAR10')

        to structured graphs of nodes and edges with default configurations.
        """
        prompt_lower = prompt.lower()
        
        nodes = []
        edges = []
        name = ""
        description = ""
        
        # Helper to generate nodes and sequentially link them
        def build_sequential_graph(layer_specs: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            node_list = []
            edge_list = []
            prev_id = None
            
            y_pos = 50
            for spec in layer_specs:
                node_id = str(uuid.uuid4())
                node_list.append({
                    "id": node_id,
                    "type": spec["type"],
                    "label": spec["label"],
                    "config": spec.get("config", {}),
                    "position_x": 250,
                    "position_y": y_pos,
                    "input_shape": spec.get("input_shape"),
                    "output_shape": spec.get("output_shape")
                })
                if prev_id:
                    edge_list.append({
                        "id": str(uuid.uuid4()),
                        "from_node_id": prev_id,
                        "to_node_id": node_id,
                        "input_shape": None,
                        "output_shape": None
                    })
                prev_id = node_id
                y_pos += 120
                
            return node_list, edge_list

        if "cifar10" in prompt_lower or "cifar" in prompt_lower or "cifar-10" in prompt_lower:
            name = "CIFAR-10 Image Classifier"
            description = "A standard Convolutional Neural Network for CIFAR-10 classification (10 classes, 32x32 RGB images)."
            specs = [
                {"type": "Input", "label": "Input Image", "config": {"shape": [None, 3, 32, 32]}, "input_shape": None, "output_shape": [None, 3, 32, 32]},
                {"type": "Conv2D", "label": "Conv2D Layer 1", "config": {"filters": 32, "kernel_size": 3, "activation": "relu"}, "input_shape": [None, 3, 32, 32], "output_shape": [None, 32, 30, 30]},
                {"type": "MaxPool2D", "label": "MaxPool2D Layer 1", "config": {"pool_size": 2}, "input_shape": [None, 32, 30, 30], "output_shape": [None, 32, 15, 15]},
                {"type": "Flatten", "label": "Flatten Layer", "config": {}, "input_shape": [None, 32, 15, 15], "output_shape": [None, 7200]},
                {"type": "Dense", "label": "Dense Layer 1", "config": {"units": 128, "activation": "relu"}, "input_shape": [None, 7200], "output_shape": [None, 128]},
                {"type": "Dense", "label": "Output Class Projections", "config": {"units": 10, "activation": "softmax"}, "input_shape": [None, 128], "output_shape": [None, 10]}
            ]
            nodes, edges = build_sequential_graph(specs)
            
        elif "mnist" in prompt_lower or "digit" in prompt_lower:
            name = "MNIST Digit Classifier"
            description = "A simple Convolutional Neural Network for MNIST digit recognition (10 classes, 28x28 grayscale images)."
            specs = [
                {"type": "Input", "label": "Input Digit", "config": {"shape": [None, 1, 28, 28]}, "input_shape": None, "output_shape": [None, 1, 28, 28]},
                {"type": "Conv2D", "label": "Conv2D Layer 1", "config": {"filters": 16, "kernel_size": 3, "activation": "relu"}, "input_shape": [None, 1, 28, 28], "output_shape": [None, 16, 26, 26]},
                {"type": "MaxPool2D", "label": "MaxPool2D Layer 1", "config": {"pool_size": 2}, "input_shape": [None, 16, 26, 26], "output_shape": [None, 16, 13, 13]},
                {"type": "Flatten", "label": "Flatten Layer", "config": {}, "input_shape": [None, 16, 13, 13], "output_shape": [None, 2704]},
                {"type": "Dense", "label": "Dense Layer 1", "config": {"units": 64, "activation": "relu"}, "input_shape": [None, 2704], "output_shape": [None, 64]},
                {"type": "Dense", "label": "Output Class Projections", "config": {"units": 10, "activation": "softmax"}, "input_shape": [None, 64], "output_shape": [None, 10]}
            ]
            nodes, edges = build_sequential_graph(specs)
            
        elif "transformer" in prompt_lower or "nlp" in prompt_lower or "text" in prompt_lower:
            name = "NLP Sentiment Transformer"
            description = "A sequence-processing block featuring embedding and self-attention for text classification tasks."
            specs = [
                {"type": "Input", "label": "Token Sequences", "config": {"shape": [None, 128]}, "input_shape": None, "output_shape": [None, 128]},
                {"type": "Embedding", "label": "Word Embeddings", "config": {"input_dim": 10000, "output_dim": 256}, "input_shape": [None, 128], "output_shape": [None, 128, 256]},
                {"type": "MultiHeadAttention", "label": "Self Attention Block", "config": {"num_heads": 8, "embed_dim": 256}, "input_shape": [None, 128, 256], "output_shape": [None, 128, 256]},
                {"type": "Flatten", "label": "Sequence Flatten", "config": {}, "input_shape": [None, 128, 256], "output_shape": [None, 32768]},
                {"type": "Dense", "label": "Dense Layer 1", "config": {"units": 256, "activation": "relu"}, "input_shape": [None, 32768], "output_shape": [None, 256]},
                {"type": "Dense", "label": "Sentiment Out", "config": {"units": 2, "activation": "softmax"}, "input_shape": [None, 256], "output_shape": [None, 2]}
            ]
            nodes, edges = build_sequential_graph(specs)
            
        elif "autoencoder" in prompt_lower:
            name = "Dense Autoencoder"
            description = "Symmetric encoder-decoder model for dimensionality reduction and anomaly detection."
            specs = [
                {"type": "Input", "label": "Flat Inputs", "config": {"shape": [None, 784]}, "input_shape": None, "output_shape": [None, 784]},
                {"type": "Dense", "label": "Encoder Layer 1", "config": {"units": 128, "activation": "relu"}, "input_shape": [None, 784], "output_shape": [None, 128]},
                {"type": "Dense", "label": "Bottleneck Latent Space", "config": {"units": 64, "activation": "relu"}, "input_shape": [None, 128], "output_shape": [None, 64]},
                {"type": "Dense", "label": "Decoder Layer 1", "config": {"units": 128, "activation": "relu"}, "input_shape": [None, 64], "output_shape": [None, 128]},
                {"type": "Dense", "label": "Reconstruction Out", "config": {"units": 784, "activation": "sigmoid"}, "input_shape": [None, 128], "output_shape": [None, 784]}
            ]
            nodes, edges = build_sequential_graph(specs)
            
        elif "regression" in prompt_lower or "tabular" in prompt_lower or "mlp" in prompt_lower:
            name = "Tabular MLP Regressor"
            description = "A Multi-Layer Perceptron optimized for tabular feature processing and continuous target value estimation."
            specs = [
                {"type": "Input", "label": "Tabular Features", "config": {"shape": [None, 20]}, "input_shape": None, "output_shape": [None, 20]},
                {"type": "Dense", "label": "Dense Projection 1", "config": {"units": 64, "activation": "relu"}, "input_shape": [None, 20], "output_shape": [None, 64]},
                {"type": "Dense", "label": "Dense Projection 2", "config": {"units": 32, "activation": "relu"}, "input_shape": [None, 64], "output_shape": [None, 32]},
                {"type": "Dense", "label": "Regression Prediction", "config": {"units": 1, "activation": "linear"}, "input_shape": [None, 32], "output_shape": [None, 1]}
            ]
            nodes, edges = build_sequential_graph(specs)
            
        else:
            # Default fallback: Standard Image Classifier CNN
            name = "Standard CNN Template"
            description = "A generic feedforward convolutional neural network suitable for image classification."
            specs = [
                {"type": "Input", "label": "Input Image", "config": {"shape": [None, 3, 224, 224]}, "input_shape": None, "output_shape": [None, 3, 224, 224]},
                {"type": "Conv2D", "label": "Conv2D Layer 1", "config": {"filters": 32, "kernel_size": 3, "activation": "relu"}, "input_shape": [None, 3, 224, 224], "output_shape": [None, 32, 222, 222]},
                {"type": "MaxPool2D", "label": "MaxPool2D Layer 1", "config": {"pool_size": 2}, "input_shape": [None, 32, 222, 222], "output_shape": [None, 32, 111, 111]},
                {"type": "Flatten", "label": "Flatten Layer", "config": {}, "input_shape": [None, 32, 111, 111], "output_shape": [None, 394272]},
                {"type": "Dense", "label": "Dense Output", "config": {"units": 10, "activation": "softmax"}, "input_shape": [None, 394272], "output_shape": [None, 10]}
            ]
            nodes, edges = build_sequential_graph(specs)

        return {
            "name": name,
            "description": description,
            "nodes": nodes,
            "edges": edges
        }
