import os
import sys
import uuid
import time
import tempfile
import subprocess
from typing import Dict, Any
from app.services.project_service import ProjectService
from app.services.validation_service import ValidationService
from app.services.shape_inference_service import ShapeInferenceService
from app.ir.ir_graph import IRGraph
from app.codegen.generators.registry import GeneratorRegistry
from app.models.project import Project
from app.models.node import Node
from app.models.edge import Edge

class NotebookExecutionService:
    @staticmethod
    def execute_cell(db, user, project_id: str, code: str, timeout: int = 60) -> Dict[str, Any]:
        try:
            proj_uuid = uuid.UUID(project_id)
        except ValueError:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Invalid project ID format.",
                "execution_time_ms": 0
            }

        # 1. Fetch project with role-sensitive checks
        if user.role == "admin":
            project = db.query(Project).filter(Project.id == proj_uuid).first()
        else:
            project = ProjectService.get_project(db, proj_uuid, user_id=user.id)

        if not project:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Project not found.",
                "execution_time_ms": 0
            }

        # 2. Fetch nodes and edges and compile to PyTorch
        compiled_module_code = ""
        try:
            nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
            edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

            if nodes:
                # Validate, sort, and infer shapes
                sorted_nodes = ValidationService.validate_graph(nodes, edges)
                ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
                db.commit()

                # Build IR graph and optimize
                ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
                from app.services.graph_engine import GraphOptimizer
                GraphOptimizer.simplify_graph(ir_graph)

                # Generate code using registry
                compiler = GeneratorRegistry.get_generator("PyTorch")
                compiled_module_code = compiler.generate(ir_graph)
                
                # Expose ArchNetModule alias to match notebook mockup imports
                compiled_module_code += "\n\nArchNetModule = GeneratedModel\n"
            else:
                compiled_module_code = (
                    "import torch\n"
                    "import torch.nn as nn\n\n"
                    "class GeneratedModel(nn.Module):\n"
                    "    def __init__(self):\n"
                    "        super().__init__()\n"
                    "        self.layer = nn.Identity()\n"
                    "    def forward(self, x):\n"
                    "        return self.layer(x)\n\n"
                    "ArchNetModule = GeneratedModel\n"
                )
        except Exception as e:
            compiled_module_code = (
                "import torch\n"
                "import torch.nn as nn\n\n"
                "class GeneratedModel(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.layer = nn.Identity()\n"
                "    def forward(self, x):\n"
                "        raise RuntimeError('Model compilation failed: " + str(e).replace("'", "\\'") + "')\n\n"
                "ArchNetModule = GeneratedModel\n"
            )

        # 3. Create sandboxed scratch directory
        scratch_dir = tempfile.mkdtemp()
        module_path = os.path.join(scratch_dir, "archnet_module.py")
        cell_path = os.path.join(scratch_dir, "cell_code.py")

        try:
            with open(module_path, "w", encoding="utf-8") as f:
                f.write(compiled_module_code)

            with open(cell_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Set execution environment PATH and pythonpath
            env = os.environ.copy()
            env["PYTHONPATH"] = scratch_dir + os.pathsep + env.get("PYTHONPATH", "")

            start_time = time.time()
            result = subprocess.run(
                [sys.executable, cell_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=scratch_dir,
                env=env
            )
            end_time = time.time()
            execution_time_ms = int((end_time - start_time) * 1000)

            # Cleanup files
            for fpath in [module_path, cell_path]:
                if os.path.exists(fpath):
                    os.remove(fpath)
            if os.path.exists(scratch_dir):
                os.rmdir(scratch_dir)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time_ms": execution_time_ms
            }

        except subprocess.TimeoutExpired:
            # Cleanup files
            for fpath in [module_path, cell_path]:
                if os.path.exists(fpath):
                    os.remove(fpath)
            if os.path.exists(scratch_dir):
                os.rmdir(scratch_dir)
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out. Exceeded the {timeout}-second sandbox limits.",
                "execution_time_ms": timeout * 1000
            }
        except Exception as e:
            # Cleanup files
            for fpath in [module_path, cell_path]:
                if os.path.exists(fpath):
                    os.remove(fpath)
            if os.path.exists(scratch_dir):
                try:
                    os.rmdir(scratch_dir)
                except Exception:
                    pass
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution engine error: {str(e)}",
                "execution_time_ms": 0
            }
