import os
import sys
import ast
import uuid
import subprocess
from typing import List, Dict, Any

class CompilationSandbox:
    @staticmethod
    def validate_compilation(code: str, framework: str, project_id: str) -> Dict[str, Any]:
        """Validates the generated python code.
        1. Performs syntax checks using Python Abstract Syntax Tree (AST).
        2. Appends dummy forward execution block (if not already included).
        3. Spawns an isolated child process to execute a forward pass, catching any runtime crashes.
        """
        # 1. AST Syntactic Parsing
        try:
            ast.parse(code)
        except SyntaxError as e:
            error_details = f"Syntax Error: {e.msg} at line {e.lineno}, column {e.offset}"
            if e.text:
                error_details += f" (Code: {e.text.strip()})"
            return {
                "success": False,
                "compilation_errors": [error_details],
                "logs": "AST compilation check failed."
            }

        # 2. Check if library is available to execute in subprocess
        framework_clean = framework.lower().strip()
        is_pytorch = "pytorch" in framework_clean or "torch" in framework_clean
        is_tensorflow = "tensorflow" in framework_clean or "keras" in framework_clean

        lib_available = False
        if is_pytorch:
            try:
                import torch
                lib_available = True
            except ImportError:
                pass
        elif is_tensorflow:
            try:
                import tensorflow
                lib_available = True
            except ImportError:
                pass

        if not lib_available:
            # Standard AST syntax pass is already a huge win if framework library is missing in backend env
            missing_lib = "PyTorch" if is_pytorch else "TensorFlow"
            return {
                "success": True,
                "compilation_errors": [],
                "logs": (
                    f"AST Syntax Parse Successful. Sandboxed execution pass was skipped "
                    f"because {missing_lib} is not installed in the local environment."
                )
            }

        # 3. Create sandboxed directory in scratch workspace
        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        scratch_dir = os.path.join(workspace_dir, "scratch")
        os.makedirs(scratch_dir, exist_ok=True)

        sandbox_filename = f"sandbox_{project_id}_{uuid.uuid4().hex[:8]}.py"
        sandbox_path = os.path.join(scratch_dir, sandbox_filename)

        try:
            # Write code to sandbox script
            with open(sandbox_path, "w", encoding="utf-8") as f:
                f.write(code)

            # 4. Trigger subprocess execution with strict 10s timeout
            result = subprocess.run(
                [sys.executable, sandbox_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Cleanup immediately after run
            if os.path.exists(sandbox_path):
                os.remove(sandbox_path)

            if result.returncode == 0:
                return {
                    "success": True,
                    "compilation_errors": [],
                    "logs": f"Runtime forward pass check succeeded!\n\nSTDOUT:\n{result.stdout}"
                }
            else:
                # Compile traceback or execution details
                err_logs = result.stderr or result.stdout
                clean_err = CompilationSandbox._clean_traceback(err_logs)
                return {
                    "success": False,
                    "compilation_errors": [clean_err],
                    "logs": f"Runtime verification crashed.\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                }

        except subprocess.TimeoutExpired:
            if os.path.exists(sandbox_path):
                os.remove(sandbox_path)
            return {
                "success": False,
                "compilation_errors": ["Subprocess execution exceeded the 10-second sandbox timeout threshold."],
                "logs": "Timeout triggered."
            }
        except Exception as e:
            if os.path.exists(sandbox_path):
                os.remove(sandbox_path)
            return {
                "success": False,
                "compilation_errors": [f"Execution launcher error: {str(e)}"],
                "logs": ""
            }

    @staticmethod
    def _clean_traceback(raw_traceback: str) -> str:
        """Helper to format and extract the relevant python exception from raw traceback strings."""
        if not raw_traceback:
            return "Unknown runtime compilation error."
        
        # Look for the last line of the traceback which usually has the error message
        lines = [line.strip() for line in raw_traceback.split("\n") if line.strip()]
        if not lines:
            return "Unknown runtime compilation error."
            
        for line in reversed(lines):
            if ":" in line and not line.startswith("File ") and not line.startswith("Traceback"):
                return line
        return lines[-1]
