import os
import json
import logging
import re
import httpx
from typing import Any
from app.config.settings import settings

logger = logging.getLogger("mlbuilder.copilot.graph_agent")

class CopilotGraphAgent:
    @staticmethod
    def execute_agent(system_prompt: str, user_prompt: str, json_response: bool = False) -> str:
        """Orchestrates model calls. Checks for Groq, Gemini, and OpenAI API keys in settings/env.
        If none are present, falls back to a deterministic rule-based mock engine.
        """
        # Determine active LLM credentials
        groq_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        gemini_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        openai_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")

        if groq_key:
            try:
                logger.info("Calling Groq API...")
                return CopilotGraphAgent._call_groq(groq_key, system_prompt, user_prompt)
            except Exception as e:
                logger.error(f"Groq API call failed: {e}. Falling back...")

        if gemini_key:
            try:
                logger.info("Calling Gemini API...")
                return CopilotGraphAgent._call_gemini(gemini_key, system_prompt, user_prompt)
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}. Falling back...")

        if openai_key:
            try:
                logger.info("Calling OpenAI API...")
                return CopilotGraphAgent._call_openai(openai_key, system_prompt, user_prompt)
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}. Falling back...")

        # Fallback to Mock Engine
        logger.info("No LLM API keys found or model calls failed. Running deterministic mock engine fallback.")
        return CopilotGraphAgent._mock_fallback(system_prompt, user_prompt, json_response)

    @staticmethod
    def _call_groq(api_key: str, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _call_gemini(api_key: str, system_prompt: str, user_prompt: str) -> str:
        # We can construct the content in a unified format
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
            }],
            "generationConfig": {
                "temperature": 0.1
            }
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    @staticmethod
    def _call_openai(api_key: str, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _mock_fallback(system_prompt: str, user_prompt: str, json_response: bool) -> str:
        """A deterministic mockup LLM simulator for testing and local runs without internet/keys."""
        user_prompt_lower = user_prompt.lower()

        # 1. GENERATION SCENARIOS
        if "generate a neural network graph" in system_prompt.lower() or "build" in user_prompt_lower:
            if "resnet50" in user_prompt_lower or "resnet" in user_prompt_lower:
                return json.dumps({
                    "nodes": [
                        {"id": "input_1", "type": "Input", "label": "Input", "config": {"shape": [None, 3, 224, 224]}},
                        {"id": "conv_1", "type": "Conv2D", "label": "Conv1", "config": {"filters": 64, "kernel_size": 7, "stride": 2}},
                        {"id": "bn_1", "type": "BatchNorm2D", "label": "Bn1", "config": {}},
                        {"id": "relu_1", "type": "ReLU", "label": "Relu1", "config": {}},
                        {"id": "maxpool_1", "type": "MaxPool2D", "label": "Maxpool1", "config": {"pool_size": 3, "stride": 2}},
                        {"id": "flatten_1", "type": "Flatten", "label": "Flatten", "config": {}},
                        {"id": "fc_1", "type": "Dense", "label": "Fc", "config": {"units": 1000}}
                    ],
                    "edges": [
                        {"from_node_id": "input_1", "to_node_id": "conv_1"},
                        {"from_node_id": "conv_1", "to_node_id": "bn_1"},
                        {"from_node_id": "bn_1", "to_node_id": "relu_1"},
                        {"from_node_id": "relu_1", "to_node_id": "maxpool_1"},
                        {"from_node_id": "maxpool_1", "to_node_id": "flatten_1"},
                        {"from_node_id": "flatten_1", "to_node_id": "fc_1"}
                    ]
                })
            elif "cnn" in user_prompt_lower:
                return json.dumps({
                    "nodes": [
                        {"id": "input_1", "type": "Input", "label": "Input", "config": {"shape": [None, 3, 32, 32]}},
                        {"id": "conv_1", "type": "Conv2D", "label": "Conv1", "config": {"filters": 32, "kernel_size": 3}},
                        {"id": "bn_1", "type": "BatchNorm2D", "label": "Bn1", "config": {}},
                        {"id": "maxpool_1", "type": "MaxPool2D", "label": "Maxpool1", "config": {"pool_size": 2}},
                        {"id": "flatten_1", "type": "Flatten", "label": "Flatten", "config": {}},
                        {"id": "fc_1", "type": "Dense", "label": "Fc", "config": {"units": 10}}
                    ],
                    "edges": [
                        {"from_node_id": "input_1", "to_node_id": "conv_1"},
                        {"from_node_id": "conv_1", "to_node_id": "bn_1"},
                        {"from_node_id": "bn_1", "to_node_id": "maxpool_1"},
                        {"from_node_id": "maxpool_1", "to_node_id": "flatten_1"},
                        {"from_node_id": "flatten_1", "to_node_id": "fc_1"}
                    ]
                })
            else:
                # General default MLP/Linear model
                return json.dumps({
                    "nodes": [
                        {"id": "input_1", "type": "Input", "label": "Input", "config": {"shape": [None, 100]}},
                        {"id": "fc_1", "type": "Dense", "label": "Fc1", "config": {"units": 64}},
                        {"id": "fc_2", "type": "Dense", "label": "Fc2", "config": {"units": 10}}
                    ],
                    "edges": [
                        {"from_node_id": "input_1", "to_node_id": "fc_1"},
                        {"from_node_id": "fc_1", "to_node_id": "fc_2"}
                    ]
                })

        # 2. MODIFICATION SCENARIOS
        if "modify the existing neural network graph" in system_prompt.lower():
            # Parse the context graph structure if present
            # We will search for nodes and recreate them, adding the mutation
            nodes = []
            edges = []
            
            node_matches = re.findall(r"- Node ID: ([^\n]+)\n\s+Type: ([^\n]+)\n\s+Label: ([^\n]+)", user_prompt)
            for nid, ntype, nlabel in node_matches:
                nodes.append({"id": nid.strip(), "type": ntype.strip(), "label": nlabel.strip(), "config": {}})
            
            edge_matches = re.findall(r"- ([a-zA-Z0-9_\-]+) -> ([a-zA-Z0-9_\-]+)", user_prompt)
            for src, dst in edge_matches:
                edges.append({"from_node_id": src, "to_node_id": dst})

            if not nodes:
                # If regex fails, fallback to default MLP and add dropout
                nodes = [
                    {"id": "input_1", "type": "Input", "label": "Input", "config": {"shape": [None, 100]}},
                    {"id": "fc_1", "type": "Dense", "label": "Fc1", "config": {"units": 64}},
                    {"id": "fc_2", "type": "Dense", "label": "Fc2", "config": {"units": 10}}
                ]
                edges = [
                    {"from_node_id": "input_1", "to_node_id": "fc_1"},
                    {"from_node_id": "fc_1", "to_node_id": "fc_2"}
                ]

            if "dropout" in user_prompt_lower:
                # Insert Dropout layer after the first node
                dropout_id = "dropout_new"
                nodes.append({"id": dropout_id, "type": "Dropout", "label": "Dropout", "config": {"rate": 0.5}})
                
                # Reconnect first node output to dropout, and dropout to the rest
                if edges:
                    first_edge = edges[0]
                    orig_from = first_edge["from_node_id"]
                    orig_to = first_edge["to_node_id"]
                    edges[0] = {"from_node_id": orig_from, "to_node_id": dropout_id}
                    edges.append({"from_node_id": dropout_id, "to_node_id": orig_to})
                else:
                    edges.append({"from_node_id": nodes[0]["id"], "to_node_id": dropout_id})

            elif "batchnorm" in user_prompt_lower or "batch_norm" in user_prompt_lower:
                bn_id = "bn_new"
                nodes.append({"id": bn_id, "type": "BatchNorm2D", "label": "BatchNorm", "config": {}})
                # Insert BN after first Conv2D node (or first node)
                target_node = nodes[0]["id"]
                for n in nodes:
                    if n["type"] == "Conv2D":
                        target_node = n["id"]
                        break
                # Find edges leaving target_node
                new_edges = []
                for e in edges:
                    if e["from_node_id"] == target_node:
                        new_edges.append({"from_node_id": bn_id, "to_node_id": e["to_node_id"]})
                        e["to_node_id"] = bn_id
                edges.extend(new_edges)

            return json.dumps({"nodes": nodes, "edges": edges})

        # 3. REFACTORING SCENARIOS
        if "suggest specific optimizations" in system_prompt.lower():
            suggestions = [
                {
                    "category": "Memory Optimization",
                    "description": "Insert MaxPooling layers to reduce spatial dimension and activation memory footprint.",
                    "action": {
                        "type": "Insert Pooling",
                        "params": {"node_id": "conv_1", "new_layer_type": "MaxPool2D", "config": {"pool_size": 2}}
                    }
                },
                {
                    "category": "Latency Optimization",
                    "description": "Add Batch Normalization layers to improve training convergence speed and stabilize gradients.",
                    "action": {
                        "type": "Insert BatchNorm",
                        "params": {"node_id": "conv_1", "new_layer_type": "BatchNorm2D", "config": {}}
                    }
                },
                {
                    "category": "Parameter Reduction",
                    "description": "Reduce large Dense classification head parameters by replacing wide linear layers.",
                    "action": {
                        "type": "Replace Layers",
                        "params": {"node_id": "fc_1", "new_layer_type": "Dense", "config": {"units": 10}}
                    }
                }
            ]
            return json.dumps(suggestions)

        # 4. EXPLANATION SCENARIO
        return (
            "# Architecture Design Explanation\n\n"
            "This model architecture is structured as a standard deep learning model pipeline.\n\n"
            "## Key Components\n"
            "- **Input Layer**: Defines the entry tensor dimensions.\n"
            "- **Feature Extraction Blocks**: Utilizes 2D Convolutions combined with Batch Normalization and Activation to construct intermediate feature representations.\n"
            "- **Classification Head**: Flattens the feature map and maps it to the target dimensions using Fully Connected (Dense) layers.\n\n"
            "## Recommendations\n"
            "1. Consider adding `Dropout` to mitigate overfitting in fully-connected layers.\n"
            "2. Ensure activation shapes match the backend configurations and compile cleanly."
        )

    @staticmethod
    def parse_json_content(content: str) -> Any:
        """Cleans and parses a JSON string returned by the LLM (removing markdown backticks/formatting)."""
        cleaned = content.strip()
        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            # Strip start fence
            cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
            # Strip end fence
            cleaned = re.sub(r"\n```$", "", cleaned)
            cleaned = cleaned.strip()
            
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON. Content: {content}. Error: {e}")
            raise ValueError(f"LLM did not output a valid JSON structure. Raw response: {content}")
