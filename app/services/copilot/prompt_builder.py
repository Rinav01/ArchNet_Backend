class CopilotPromptBuilder:
    @staticmethod
    def build_generation_prompt(prompt: str, framework: str) -> tuple[str, str]:
        """Builds system and user prompts to generate a neural network graph."""
        system_prompt = (
            "You are an AI deep learning architect assistant. Your task is to construct neural network graphs in a specific JSON format.\n"
            f"Target framework: {framework}.\n"
            "You must output ONLY a valid JSON object matching this schema. Do NOT wrap it in any Markdown code blocks or add any comments or text before or after the JSON:\n"
            "{\n"
            '  "nodes": [\n'
            "    {\n"
            '      "id": "string (unique temporary ID, e.g., input_1, conv_1, dense_1)",\n'
            '      "type": "string (one of: Input, Conv2D, Dense, MaxPool2D, Flatten, Add, Concatenate, BatchNorm2D, Dropout, ReLU)",\n'
            '      "label": "string (human readable label, e.g., Conv1, InputLayer)",\n'
            '      "config": { ... layer parameters like filters, kernel_size, units, rate, activation, etc. ... }\n'
            "    }\n"
            '  ],\n'
            '  "edges": [\n'
            "    {\n"
            '      "from_node_id": "string (matching a node ID)",\n'
            '      "to_node_id": "string (matching a node ID)"\n'
            "    }\n"
            '  ]\n'
            "}\n"
            "Ensure the graph represents a valid, connected neural network flowing from Input to final output layer. The output must be pure, parsable JSON."
        )
        user_prompt = f"Generate a neural network graph based on this prompt: '{prompt}'."
        return system_prompt, user_prompt

    @staticmethod
    def build_modification_prompt(prompt: str, graph_context: str) -> tuple[str, str]:
        """Builds prompts to edit an existing neural network graph."""
        system_prompt = (
            "You are an AI deep learning architect assistant. Your task is to modify the existing neural network graph based on the user's instructions.\n"
            "Analyze the current graph state provided in the context, apply the modifications, and output the complete, updated graph in the exact JSON format specified below.\n"
            "Do NOT output any comments, markdown fences, or text before/after. Output ONLY valid, parsable JSON:\n"
            "{\n"
            '  "nodes": [\n'
            "    {\n"
            '      "id": "string (reuse existing node IDs or create new ones if inserting)",\n'
            '      "type": "string (e.g., Input, Conv2D, Dense, MaxPool2D, Flatten, Add, Concatenate, BatchNorm2D, Dropout, ReLU)",\n'
            '      "label": "string",\n'
            '      "config": { ... }\n'
            "    }\n"
            '  ],\n'
            '  "edges": [\n'
            "    {\n"
            '      "from_node_id": "string",\n'
            '      "to_node_id": "string"\n'
            "    }\n"
            '  ]\n'
            "}\n"
            "Be careful to preserve existing node IDs and connections unless they are explicitly modified, replaced, or deleted."
        )
        user_prompt = f"Current Graph State:\n{graph_context}\n\nUser Modification Instruction: '{prompt}'."
        return system_prompt, user_prompt

    @staticmethod
    def build_explanation_prompt(graph_context: str) -> tuple[str, str]:
        """Builds prompts to explain the network architecture."""
        system_prompt = (
            "You are an expert deep learning engineer. Explain the given neural network architecture clearly.\n"
            "Provide a comprehensive, professional breakdown in Markdown format. Cover:\n"
            "- Architecture overview and purpose\n"
            "- Design choices, key layers, and parameters\n"
            "- Layer dimensions, flow of data, and connectivity\n"
            "- Potential bottlenecks or design observations"
        )
        user_prompt = f"Graph Context:\n{graph_context}"
        return system_prompt, user_prompt

    @staticmethod
    def build_refactoring_prompt(graph_context: str) -> tuple[str, str]:
        """Builds prompts to suggest optimizations and refactoring actions."""
        system_prompt = (
            "You are an expert AI deep learning optimization assistant. Your task is to analyze the neural network graph and suggest specific optimizations.\n"
            "Analyze the graph context for optimizations in these categories:\n"
            "- Memory Optimization\n"
            "- Latency Optimization\n"
            "- Parameter Reduction\n"
            "\n"
            "For each suggestion, you can optionally define a structured AI Action. Supported actions are:\n"
            "- Insert Pooling\n"
            "- Insert BatchNorm\n"
            "- Replace Layers\n"
            "- Remove Bottlenecks\n"
            "\n"
            "You must output ONLY a valid JSON list of suggestions matching the schema below. Do NOT output any explanation text or markdown wrappers. Output ONLY JSON:\n"
            "[\n"
            "  {\n"
            '    "category": "string (Memory Optimization, Latency Optimization, or Parameter Reduction)",\n'
            '    "description": "string (detailed rationale)",\n'
            '    "action": {\n'
            '      "type": "string (Insert Pooling, Insert BatchNorm, Replace Layers, or Remove Bottlenecks)",\n'
            '      "params": {\n'
            '        "node_id": "string (ID of the node target, or null)",\n'
            '        "new_layer_type": "string (e.g., BatchNorm2D, MaxPool2D, Conv2D, or null)",\n'
            '        "config": { ... optional params dict ... }\n'
            "      }\n"
            "    }\n"
            "  }\n"
            "]"
        )
        user_prompt = f"Graph Context:\n{graph_context}"
        return system_prompt, user_prompt
