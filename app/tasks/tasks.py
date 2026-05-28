import uuid
from celery import shared_task
from app.tasks.celery_app import celery_app
from app.config.database import SessionLocal
from app.models.project import Project
from app.models.node import Node
from app.models.edge import Edge
from app.services.validation_service import ValidationService
from app.services.shape_inference_service import ShapeInferenceService
from app.services.event_dispatcher import EventDispatcher
from app.services.compilation_sandbox import CompilationSandbox
from app.ir.ir_graph import IRGraph
from app.services.graph_engine import GraphOptimizer

@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def async_compile_and_validate(self, project_id_str: str, user_id_str: str):
    """Asynchronously validates a canvas architecture, compiles the model code,
    and runs a sandboxed subprocess verification forward pass.
    Publishes event updates over WebSockets/Redis PubSub.
    """
    proj_uuid = uuid.UUID(project_id_str)
    user_uuid = uuid.UUID(user_id_str)
    
    # 1. Fire CompilationStarted event
    EventDispatcher.dispatch_compilation_started(proj_uuid, self.request.id)

    db = SessionLocal()
    try:
        # Retrieve project and canvas components
        project = db.query(Project).filter(Project.id == proj_uuid, Project.user_id == user_uuid).first()
        if not project:
            err = "Project not found or unauthorized access."
            EventDispatcher.dispatch_validation_failed(proj_uuid, [err])
            return {"success": False, "errors": [err]}

        nodes = db.query(Node).filter(Node.project_id == proj_uuid).all()
        edges = db.query(Edge).filter(Edge.project_id == proj_uuid).all()

        semantic_errors = []
        compatibility_errors = []
        generated_code = ""
        sorted_nodes = []

        # 2. Graph Topo Sorting and Shapes Inference
        try:
            sorted_nodes = ValidationService.validate_graph(nodes, edges)
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
            db.commit() # Save computed shapes
            
            # Semantic Checks
            semantic_errors = ValidationService.validate_semantics(sorted_nodes, edges)
        except Exception as dag_err:
            semantic_errors = [str(dag_err)]

        if semantic_errors:
            EventDispatcher.dispatch_validation_failed(proj_uuid, semantic_errors)
            return {"success": False, "errors": semantic_errors}

        # 3. Code Generation and Sandbox Subprocess Execution
        try:
            # Compatibility warning checks
            compatibility_errors = ValidationService.validate_framework_compatibility(
                sorted_nodes, edges, project.framework
            )

            # Build intermediate representations
            ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
            GraphOptimizer.simplify_graph(ir_graph)

            framework_str = project.framework.lower().strip()
            if "pytorch" in framework_str or "torch" in framework_str:
                from app.codegen.pytorch.generator import PyTorchCompiler
                compiler = PyTorchCompiler()
                generated_code = compiler.compile(ir_graph)
            elif "tensorflow" in framework_str or "keras" in framework_str:
                from app.codegen.tensorflow.compiler import TensorFlowCompiler
                compiler = TensorFlowCompiler()
                generated_code = compiler.compile(ir_graph)
            else:
                generated_code = f"# Asynchronous compiler for '{project.framework}' is not supported."

            # Execute Sandbox Child Verification
            sandbox_res = CompilationSandbox.validate_compilation(
                code=generated_code,
                framework=project.framework,
                project_id=str(project.id)
            )

            if sandbox_res["success"]:
                # Success! Dispatch CompilationFinished with clean code and logs
                EventDispatcher.dispatch_compilation_finished(
                    proj_uuid, generated_code, sandbox_res["logs"]
                )
                return {
                    "success": True,
                    "generated_code": generated_code,
                    "logs": sandbox_res["logs"],
                    "compatibility_warnings": compatibility_errors
                }
            else:
                # Sandbox fail: Dispatch ValidationFailed with the sandboxed tracebacks
                EventDispatcher.dispatch_validation_failed(
                    proj_uuid, sandbox_res["compilation_errors"]
                )
                return {
                    "success": False,
                    "errors": sandbox_res["compilation_errors"],
                    "logs": sandbox_res["logs"]
                }

        except Exception as compile_err:
            # Fatal error inside generator/optimizer
            errs = [f"Internal compiler crash: {str(compile_err)}"]
            EventDispatcher.dispatch_validation_failed(proj_uuid, errs)
            return {"success": False, "errors": errs}

    except Exception as exc:
        # DB connection crashes or worker problems: retry task
        try:
            self.retry(exc=exc)
        except Exception as retry_exc:
            errs = [f"Task orchestration crash: {str(exc)}"]
            EventDispatcher.dispatch_validation_failed(proj_uuid, errs)
            return {"success": False, "errors": errs}
    finally:
        db.close()
