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
            
        elif "sentiment" in prompt_lower:
            name = "Sentiment Classifier"
            description = "A sequence classification network featuring Word Embeddings and LSTM for sentiment analysis."
            specs = [
                {"type": "Input", "label": "Token Sequences", "config": {"shape": [None, 128]}, "input_shape": None, "output_shape": [None, 128]},
                {"type": "Embedding", "label": "Word Embeddings", "config": {"input_dim": 10000, "output_dim": 256}, "input_shape": [None, 128], "output_shape": [None, 128, 256]},
                {"type": "LSTM", "label": "LSTM Sequence Processing", "config": {"hidden_size": 128, "return_sequences": False}, "input_shape": [None, 128, 256], "output_shape": [None, 128]},
                {"type": "Dense", "label": "Sentiment Output Projection", "config": {"units": 2, "activation": "softmax"}, "input_shape": [None, 128], "output_shape": [None, 2]}
            ]
            nodes, edges = build_sequential_graph(specs)

        elif "text classifier" in prompt_lower or "text_classifier" in prompt_lower or "text-classifier" in prompt_lower:
            name = "Text Classifier"
            description = "Multi-class text classifier using BiLSTM and GRU sequence processing layers."
            specs = [
                {"type": "Input", "label": "Text Tokens", "config": {"shape": [None, 128]}, "input_shape": None, "output_shape": [None, 128]},
                {"type": "Embedding", "label": "Token Embeddings", "config": {"input_dim": 10000, "output_dim": 128}, "input_shape": [None, 128], "output_shape": [None, 128, 128]},
                {"type": "BiLSTM", "label": "Bidirectional LSTM", "config": {"hidden_size": 64, "return_sequences": True}, "input_shape": [None, 128, 128], "output_shape": [None, 128, 128]},
                {"type": "GRU", "label": "GRU Layer", "config": {"hidden_size": 64, "return_sequences": False}, "input_shape": [None, 128, 128], "output_shape": [None, 64]},
                {"type": "Dense", "label": "Output Projections", "config": {"units": 5, "activation": "softmax"}, "input_shape": [None, 64], "output_shape": [None, 5]}
            ]
            nodes, edges = build_sequential_graph(specs)

        elif "seq2seq" in prompt_lower:
            name = "Seq2Seq"
            description = "Sequence-to-Sequence (Seq2Seq) model layout with paired Encoder and Decoder LSTM sequence layers."
            specs = [
                {"type": "Input", "label": "Source Sequences", "config": {"shape": [None, 80]}, "input_shape": None, "output_shape": [None, 80]},
                {"type": "Embedding", "label": "Source Embeddings", "config": {"input_dim": 5000, "output_dim": 256}, "input_shape": [None, 80], "output_shape": [None, 80, 256]},
                {"type": "LSTM", "label": "Encoder LSTM", "config": {"hidden_size": 256, "return_sequences": True}, "input_shape": [None, 80, 256], "output_shape": [None, 80, 256]},
                {"type": "LSTM", "label": "Decoder LSTM", "config": {"hidden_size": 256, "return_sequences": True}, "input_shape": [None, 80, 256], "output_shape": [None, 80, 256]},
                {"type": "Dense", "label": "Target Vocabulary Dense", "config": {"units": 5000, "activation": "softmax"}, "input_shape": [None, 80, 256], "output_shape": [None, 80, 5000]}
            ]
            nodes, edges = build_sequential_graph(specs)

        elif "mini-bert" in prompt_lower or "mini_bert" in prompt_lower or "mini bert" in prompt_lower:
            name = "Mini-BERT"
            description = "A compact bidirectional Transformer (Mini-BERT) layout for masked token sequence encoding."
            specs = [
                {"type": "Input", "label": "Masked Token Inputs", "config": {"shape": [None, 128]}, "input_shape": None, "output_shape": [None, 128]},
                {"type": "Embedding", "label": "Word Embeddings", "config": {"input_dim": 10000, "output_dim": 256}, "input_shape": [None, 128], "output_shape": [None, 128, 256]},
                {"type": "PositionalEncoding", "label": "Positional Encodings", "config": {"embedding_dim": 256, "max_len": 128}, "input_shape": [None, 128, 256], "output_shape": [None, 128, 256]},
                {"type": "LayerNorm", "label": "LayerNorm 1", "config": {}, "input_shape": [None, 128, 256], "output_shape": [None, 128, 256]},
                {"type": "TransformerBlock", "label": "Encoder Block 1", "config": {"num_heads": 4, "embed_dim": 256}, "input_shape": [None, 128, 256], "output_shape": [None, 128, 256]},
                {"type": "TransformerBlock", "label": "Encoder Block 2", "config": {"num_heads": 4, "embed_dim": 256}, "input_shape": [None, 128, 256], "output_shape": [None, 128, 256]},
                {"type": "Dense", "label": "Vocabulary Projection Output", "config": {"units": 10000, "activation": "softmax"}, "input_shape": [None, 128, 256], "output_shape": [None, 128, 10000]}
            ]
            nodes, edges = build_sequential_graph(specs)

        elif "mini-gpt" in prompt_lower or "mini_gpt" in prompt_lower or "mini gpt" in prompt_lower:
            name = "Mini-GPT"
            description = "A compact autoregressive Transformer (Mini-GPT) layout for next-token generation."
            specs = [
                {"type": "Input", "label": "Context Token Inputs", "config": {"shape": [None, 128]}, "input_shape": None, "output_shape": [None, 128]},
                {"type": "Embedding", "label": "Token Embeddings", "config": {"input_dim": 10000, "output_dim": 256}, "input_shape": [None, 128], "output_shape": [None, 128, 256]},
                {"type": "PositionalEncoding", "label": "Autoregressive Positionals", "config": {"embedding_dim": 256, "max_len": 128}, "input_shape": [None, 128, 256], "output_shape": [None, 128, 256]},
                {"type": "TransformerBlock", "label": "Decoder Block 1", "config": {"num_heads": 4, "embed_dim": 256}, "input_shape": [None, 128, 256], "output_shape": [None, 128, 256]},
                {"type": "TransformerBlock", "label": "Decoder Block 2", "config": {"num_heads": 4, "embed_dim": 256}, "input_shape": [None, 128, 256], "output_shape": [None, 128, 256]},
                {"type": "Dense", "label": "Next Token Predictions", "config": {"units": 10000, "activation": "softmax"}, "input_shape": [None, 128, 256], "output_shape": [None, 128, 10000]}
            ]
            nodes, edges = build_sequential_graph(specs)

        elif "transformer encoder" in prompt_lower or "transformer_encoder" in prompt_lower:
            name = "Transformer Encoder"
            description = "Standard multi-head Transformer Encoder stack optimized for representation learning."
            specs = [
                {"type": "Input", "label": "Input Sequences", "config": {"shape": [None, 64]}, "input_shape": None, "output_shape": [None, 64]},
                {"type": "Embedding", "label": "Sequence Embeddings", "config": {"input_dim": 1000, "output_dim": 512}, "input_shape": [None, 64], "output_shape": [None, 64, 512]},
                {"type": "PositionalEncoding", "label": "Positional Signatures", "config": {"embedding_dim": 512, "max_len": 64}, "input_shape": [None, 64, 512], "output_shape": [None, 64, 512]},
                {"type": "TransformerBlock", "label": "Encoder Stack", "config": {"num_heads": 8, "embed_dim": 512}, "input_shape": [None, 64, 512], "output_shape": [None, 64, 512]},
                {"type": "LayerNorm", "label": "Stack Normalization", "config": {}, "input_shape": [None, 64, 512], "output_shape": [None, 64, 512]},
                {"type": "Flatten", "label": "Flatten Sequence Output", "config": {}, "input_shape": [None, 64, 512], "output_shape": [None, 32768]},
                {"type": "Dense", "label": "Sequence Classifier Out", "config": {"units": 2, "activation": "softmax"}, "input_shape": [None, 32768], "output_shape": [None, 2]}
            ]
            nodes, edges = build_sequential_graph(specs)

        elif "resnet18" in prompt_lower or "resnet-18" in prompt_lower:
            name = "ResNet18"
            description = "Deep Residual Network (ResNet-18) variant featuring custom stem convolution and residual skip block."
            node_ids = [str(uuid.uuid4()) for _ in range(12)]
            nodes = [
                {"id": node_ids[0], "type": "Input", "label": "Input Image", "config": {"shape": [None, 3, 224, 224]}, "position_x": 250, "position_y": 50, "input_shape": None, "output_shape": [None, 3, 224, 224]},
                {"id": node_ids[1], "type": "Conv2D", "label": "Stem Conv", "config": {"filters": 64, "kernel_size": 7, "stride": 2, "padding": "same", "activation": "None"}, "position_x": 250, "position_y": 170, "input_shape": [None, 3, 224, 224], "output_shape": [None, 64, 112, 112]},
                {"id": node_ids[2], "type": "BatchNorm2D", "label": "Stem BN", "config": {}, "position_x": 250, "position_y": 290, "input_shape": [None, 64, 112, 112], "output_shape": [None, 64, 112, 112]},
                {"id": node_ids[3], "type": "MaxPool2D", "label": "Stem Pool", "config": {"pool_size": 3, "stride": 2, "padding": 1}, "position_x": 250, "position_y": 410, "input_shape": [None, 64, 112, 112], "output_shape": [None, 64, 56, 56]},
                {"id": node_ids[4], "type": "Conv2D", "label": "Res1 Conv A", "config": {"filters": 64, "kernel_size": 3, "stride": 1, "padding": "same", "activation": "ReLU"}, "position_x": 150, "position_y": 530, "input_shape": [None, 64, 56, 56], "output_shape": [None, 64, 56, 56]},
                {"id": node_ids[5], "type": "BatchNorm2D", "label": "Res1 BN A", "config": {}, "position_x": 150, "position_y": 650, "input_shape": [None, 64, 56, 56], "output_shape": [None, 64, 56, 56]},
                {"id": node_ids[6], "type": "Conv2D", "label": "Res1 Conv B", "config": {"filters": 64, "kernel_size": 3, "stride": 1, "padding": "same", "activation": "None"}, "position_x": 150, "position_y": 770, "input_shape": [None, 64, 56, 56], "output_shape": [None, 64, 56, 56]},
                {"id": node_ids[7], "type": "BatchNorm2D", "label": "Res1 BN B", "config": {}, "position_x": 150, "position_y": 890, "input_shape": [None, 64, 56, 56], "output_shape": [None, 64, 56, 56]},
                {"id": node_ids[8], "type": "ResidualAdd", "label": "Residual Add", "config": {}, "position_x": 250, "position_y": 1010, "input_shape": [[None, 64, 56, 56], [None, 64, 56, 56]], "output_shape": [None, 64, 56, 56]},
                {"id": node_ids[9], "type": "MaxPool2D", "label": "Global Average Pool", "config": {"pool_size": 56}, "position_x": 250, "position_y": 1130, "input_shape": [None, 64, 56, 56], "output_shape": [None, 64, 1, 1]},
                {"id": node_ids[10], "type": "Flatten", "label": "Flatten Features", "config": {}, "position_x": 250, "position_y": 1250, "input_shape": [None, 64, 1, 1], "output_shape": [None, 64]},
                {"id": node_ids[11], "type": "Dense", "label": "Classifier Out", "config": {"units": 10, "activation": "softmax"}, "position_x": 250, "position_y": 1370, "input_shape": [None, 64], "output_shape": [None, 10]}
            ]
            edges = [
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[0], "to_node_id": node_ids[1]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[1], "to_node_id": node_ids[2]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[2], "to_node_id": node_ids[3]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[3], "to_node_id": node_ids[4]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[4], "to_node_id": node_ids[5]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[5], "to_node_id": node_ids[6]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[6], "to_node_id": node_ids[7]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[7], "to_node_id": node_ids[8]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[3], "to_node_id": node_ids[8]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[8], "to_node_id": node_ids[9]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[9], "to_node_id": node_ids[10]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[10], "to_node_id": node_ids[11]}
            ]

        elif "u-net" in prompt_lower or "unet" in prompt_lower or "u_net" in prompt_lower:
            name = "U-Net"
            description = "Symmetric encoder-decoder skip merge network for segmentation."
            node_ids = [str(uuid.uuid4()) for _ in range(15)]
            nodes = [
                {"id": node_ids[0], "type": "Input", "label": "INPUT_IMAGE", "config": {"shape": [None, 3, 256, 256]}, "position_x": 100, "position_y": 300, "input_shape": None, "output_shape": [None, 3, 256, 256]},
                {"id": node_ids[1], "type": "Conv2D", "label": "ENC1_CONV_64", "config": {"filters": 64, "kernel_size": 3, "stride": 1, "padding": "same", "activation": "ReLU"}, "position_x": 280, "position_y": 300, "input_shape": [None, 3, 256, 256], "output_shape": [None, 64, 256, 256]},
                {"id": node_ids[2], "type": "MaxPool2D", "label": "ENC1_MAXPOOL", "config": {"pool_size": 2}, "position_x": 460, "position_y": 300, "input_shape": [None, 64, 256, 256], "output_shape": [None, 64, 128, 128]},
                {"id": node_ids[3], "type": "Conv2D", "label": "ENC2_CONV_128", "config": {"filters": 128, "kernel_size": 3, "stride": 1, "padding": "same", "activation": "ReLU"}, "position_x": 640, "position_y": 450, "input_shape": [None, 64, 128, 128], "output_shape": [None, 128, 128, 128]},
                {"id": node_ids[4], "type": "MaxPool2D", "label": "ENC2_MAXPOOL", "config": {"pool_size": 2}, "position_x": 820, "position_y": 450, "input_shape": [None, 128, 128, 128], "output_shape": [None, 128, 64, 64]},
                {"id": node_ids[5], "type": "Conv2D", "label": "BOTTLENECK_CONV", "config": {"filters": 256, "kernel_size": 3, "stride": 1, "padding": "same", "activation": "ReLU"}, "position_x": 1000, "position_y": 600, "input_shape": [None, 128, 64, 64], "output_shape": [None, 256, 64, 64]},
                {"id": node_ids[6], "type": "Conv2D", "label": "DEC2_UP_CONV", "config": {"filters": 128, "kernel_size": 2, "stride": 1, "padding": "same", "activation": "ReLU"}, "position_x": 1180, "position_y": 450, "input_shape": [None, 256, 64, 64], "output_shape": [None, 128, 64, 64]},
                {"id": node_ids[7], "type": "Conv2D", "label": "DEC2_CONV_1", "config": {"filters": 128, "kernel_size": 3, "stride": 1, "padding": "same", "activation": "ReLU"}, "position_x": 1360, "position_y": 450, "input_shape": [[None, 128, 64, 64], [None, 128, 128, 128]], "output_shape": [None, 128, 64, 64]},
                {"id": node_ids[8], "type": "BatchNorm2D", "label": "DEC2_BN_1", "config": {}, "position_x": 1540, "position_y": 450, "input_shape": [None, 128, 64, 64], "output_shape": [None, 128, 64, 64]},
                {"id": node_ids[9], "type": "Conv2D", "label": "DEC2_CONV_2", "config": {"filters": 128, "kernel_size": 3, "stride": 1, "padding": "same", "activation": "ReLU"}, "position_x": 1720, "position_y": 450, "input_shape": [None, 128, 64, 64], "output_shape": [None, 128, 64, 64]},
                {"id": node_ids[10], "type": "BatchNorm2D", "label": "DEC2_BN_2", "config": {}, "position_x": 1900, "position_y": 450, "input_shape": [None, 128, 64, 64], "output_shape": [None, 128, 64, 64]},
                {"id": node_ids[11], "type": "Conv2D", "label": "DEC1_UP_CONV", "config": {"filters": 64, "kernel_size": 2, "stride": 1, "padding": "same", "activation": "ReLU"}, "position_x": 2080, "position_y": 300, "input_shape": [None, 128, 64, 64], "output_shape": [None, 64, 64, 64]},
                {"id": node_ids[12], "type": "Conv2D", "label": "DEC1_CONV_1", "config": {"filters": 64, "kernel_size": 3, "stride": 1, "padding": "same", "activation": "ReLU"}, "position_x": 2260, "position_y": 300, "input_shape": [[None, 64, 64, 64], [None, 64, 256, 256]], "output_shape": [None, 64, 64, 64]},
                {"id": node_ids[13], "type": "BatchNorm2D", "label": "DEC1_BN_1", "config": {}, "position_x": 2440, "position_y": 300, "input_shape": [None, 64, 64, 64], "output_shape": [None, 64, 64, 64]},
                {"id": node_ids[14], "type": "Conv2D", "label": "OUTPUT_SEG_MASK", "config": {"filters": 2, "kernel_size": 1, "stride": 1, "padding": "same", "activation": "Softmax"}, "position_x": 2620, "position_y": 300, "input_shape": [None, 64, 64, 64], "output_shape": [None, 2, 64, 64]}
            ]
            edges = [
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[0], "to_node_id": node_ids[1]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[1], "to_node_id": node_ids[2]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[2], "to_node_id": node_ids[3]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[3], "to_node_id": node_ids[4]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[4], "to_node_id": node_ids[5]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[5], "to_node_id": node_ids[6]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[6], "to_node_id": node_ids[7]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[3], "to_node_id": node_ids[7]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[7], "to_node_id": node_ids[8]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[8], "to_node_id": node_ids[9]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[9], "to_node_id": node_ids[10]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[10], "to_node_id": node_ids[11]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[11], "to_node_id": node_ids[12]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[1], "to_node_id": node_ids[12]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[12], "to_node_id": node_ids[13]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[13], "to_node_id": node_ids[14]}
            ]

        elif "vit" in prompt_lower or "vision transformer" in prompt_lower:
            name = "ViT"
            description = "Vision Transformer with patch projection and transformer encoder blocks."
            specs = [
                {"type": "Input", "label": "INPUT_IMAGE", "config": {"shape": [None, 3, 224, 224]}, "input_shape": None, "output_shape": [None, 3, 224, 224]},
                {"type": "Conv2D", "label": "PATCH_PROJECTION", "config": {"filters": 768, "kernel_size": 16, "stride": 16, "padding": "valid", "activation": "None"}, "input_shape": [None, 3, 224, 224], "output_shape": [None, 768, 14, 14]},
                {"type": "Flatten", "label": "Flatten Tokens", "config": {}, "input_shape": [None, 768, 14, 14], "output_shape": [None, 150528]},
                {"type": "Dense", "label": "Embedding Projection", "config": {"units": 768}, "input_shape": [None, 150528], "output_shape": [None, 768]},
                {"type": "PositionalEncoding", "label": "Positional Encodings", "config": {"embedding_dim": 768, "max_len": 196}, "input_shape": [None, 768], "output_shape": [None, 196, 768]},
                {"type": "TransformerBlock", "label": "Encoder Block 1", "config": {"num_heads": 12, "embed_dim": 768}, "input_shape": [None, 196, 768], "output_shape": [None, 196, 768]},
                {"type": "TransformerBlock", "label": "Encoder Block 2", "config": {"num_heads": 12, "embed_dim": 768}, "input_shape": [None, 196, 768], "output_shape": [None, 196, 768]},
                {"type": "LayerNorm", "label": "Pre-Classifier LayerNorm", "config": {}, "input_shape": [None, 196, 768], "output_shape": [None, 196, 768]},
                {"type": "Flatten", "label": "Classification Flatten", "config": {}, "input_shape": [None, 196, 768], "output_shape": [None, 150528]},
                {"type": "Dense", "label": "DENSE_CLASSIFIER", "config": {"units": 1000, "activation": "softmax"}, "input_shape": [None, 150528], "output_shape": [None, 1000]}
            ]
            nodes, edges = build_sequential_graph(specs)

        elif "gcn" in prompt_lower or "graph convolutional" in prompt_lower:
            name = "GCN"
            description = "Graph Convolutional Network (GCN) for node classification tasks."
            node_ids = [str(uuid.uuid4()) for _ in range(4)]
            nodes = [
                {"id": node_ids[0], "type": "Input", "label": "Node Features", "config": {"shape": [None, 1433]}, "position_x": 250, "position_y": 100, "input_shape": None, "output_shape": [None, 1433]},
                {"id": node_ids[1], "type": "Input", "label": "Edge Index", "config": {"shape": [2, None]}, "position_x": 250, "position_y": 300, "input_shape": None, "output_shape": [2, None]},
                {"id": node_ids[2], "type": "GCN", "label": "GCN Layer 1", "config": {"out_features": 64}, "position_x": 500, "position_y": 200, "input_shape": [[None, 1433], [2, None]], "output_shape": [None, 64]},
                {"id": node_ids[3], "type": "GCN", "label": "GCN Layer 2", "config": {"out_features": 7}, "position_x": 750, "position_y": 200, "input_shape": [[None, 64], [2, None]], "output_shape": [None, 7]}
            ]
            edges = [
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[0], "to_node_id": node_ids[2]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[1], "to_node_id": node_ids[2]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[2], "to_node_id": node_ids[3]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[1], "to_node_id": node_ids[3]}
            ]

        elif "graphsage" in prompt_lower or "graph sage" in prompt_lower or "sage" in prompt_lower:
            name = "GraphSAGE"
            description = "GraphSAGE Network featuring neighbor aggregation layers."
            node_ids = [str(uuid.uuid4()) for _ in range(4)]
            nodes = [
                {"id": node_ids[0], "type": "Input", "label": "Node Features", "config": {"shape": [None, 1433]}, "position_x": 250, "position_y": 100, "input_shape": None, "output_shape": [None, 1433]},
                {"id": node_ids[1], "type": "Input", "label": "Edge Index", "config": {"shape": [2, None]}, "position_x": 250, "position_y": 300, "input_shape": None, "output_shape": [2, None]},
                {"id": node_ids[2], "type": "GraphSAGE", "label": "SAGE Layer 1", "config": {"out_features": 64}, "position_x": 500, "position_y": 200, "input_shape": [[None, 1433], [2, None]], "output_shape": [None, 64]},
                {"id": node_ids[3], "type": "GraphSAGE", "label": "SAGE Layer 2", "config": {"out_features": 7}, "position_x": 750, "position_y": 200, "input_shape": [[None, 64], [2, None]], "output_shape": [None, 7]}
            ]
            edges = [
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[0], "to_node_id": node_ids[2]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[1], "to_node_id": node_ids[2]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[2], "to_node_id": node_ids[3]},
                {"id": str(uuid.uuid4()), "from_node_id": node_ids[1], "to_node_id": node_ids[3]}
            ]

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
