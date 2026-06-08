import uuid
import os
import torch
import numpy as np
from sqlalchemy.orm import Session
from app.models.model_artifact import ModelArtifact
from app.models.project import Project
from app.models.node import Node
from app.models.edge import Edge
from app.services.training_service import TrainingService
from app.services.validation_service import ValidationService
from app.services.shape_inference_service import ShapeInferenceService
from app.ir.ir_graph import IRGraph
from app.codegen.generators.registry import GeneratorRegistry

class InferenceService:
    @staticmethod
    def execute_prediction(db: Session, artifact_id: uuid.UUID, input_data: dict) -> dict:
        """
        Loads the ModelArtifact from database, instantiates its compiled runtime model,
        injects the saved state dict/weights, prepares the inputs, and runs forward inference.
        """
        artifact = db.query(ModelArtifact).filter(ModelArtifact.id == artifact_id).first()
        if not artifact:
            raise ValueError("Model artifact not found.")

        raw_input = input_data.get("input")
        if raw_input is None:
            raise ValueError("Missing 'input' field in input_data.")

        if artifact.framework.lower() == "pytorch":
            # 1. Compile project graph to get latest model definition
            project = db.query(Project).filter(Project.id == artifact.project_id).first()
            if not project:
                raise ValueError("Project not found.")

            nodes = db.query(Node).filter(Node.project_id == artifact.project_id).all()
            edges = db.query(Edge).filter(Edge.project_id == artifact.project_id).all()

            sorted_nodes = ValidationService.validate_graph(nodes, edges)
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)

            ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
            compiler = GeneratorRegistry.get_generator("PyTorch")
            generated_code = compiler.generate(ir_graph)

            # 2. Instantiate model and load weights
            model = TrainingService.instantiate_model(generated_code)
            if os.path.exists(artifact.artifact_path):
                model.load_state_dict(torch.load(artifact.artifact_path, map_location="cpu"))
            model.eval()

            # 3. Format input tensor
            x = torch.tensor(raw_input, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)

            # Resolve shape mismatch if needed (pad, crop, or reshape to fit the Input layer output_shape)
            try:
                with torch.no_grad():
                    output = model(x)
            except Exception:
                input_nodes = [n for n in sorted_nodes if n.type.lower() == "input"]
                if input_nodes and input_nodes[0].output_shape:
                    expected_shape = input_nodes[0].output_shape
                    # Make batch size 1
                    target_shape = [1] + [dim if dim is not None else 1 for dim in expected_shape[1:]]
                    num_el = int(np.prod(target_shape))
                    flat_input = x.flatten()
                    if flat_input.numel() < num_el:
                        padded = torch.zeros(num_el)
                        padded[:flat_input.numel()] = flat_input
                        x = padded.view(*target_shape)
                    else:
                        x = flat_input[:num_el].view(*target_shape)
                    
                    with torch.no_grad():
                        output = model(x)
                else:
                    raise

            # 4. Return Output
            pred_list = output.tolist()
            if output.shape[-1] > 1:
                prediction_val = int(output.argmax(dim=-1)[0].item())
            else:
                prediction_val = float(output[0][0].item()) if output.numel() > 0 else 0.0

            return {
                "prediction": prediction_val,
                "raw_output": pred_list[0] if len(pred_list) > 0 else pred_list
            }
        else:
            # Fallback for non-PyTorch frameworks
            val = sum(raw_input) / max(1, len(raw_input))
            return {
                "prediction": 1 if val > 0.5 else 0,
                "raw_output": [val]
            }
