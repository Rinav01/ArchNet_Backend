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
from app.config.logging import compiler_logger

@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=5,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True
)
def async_compile_and_validate(self, project_id_str: str, user_id_str: str):
    """Asynchronously validates a canvas architecture, compiles the model code,
    and runs a sandboxed subprocess verification forward pass.
    Publishes event updates over WebSockets/Redis PubSub.
    """
    proj_uuid = uuid.UUID(project_id_str)
    user_uuid = uuid.UUID(user_id_str)
    
    compiler_logger.info(
        "Compilation started",
        project_id=project_id_str,
        user_id=user_id_str,
        task_id=self.request.id
    )
    
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

            from app.codegen.generators.registry import GeneratorRegistry
            try:
                compiler = GeneratorRegistry.get_generator(project.framework)
                generated_code = compiler.generate(ir_graph)
            except ValueError:
                generated_code = f"# Asynchronous compiler for '{project.framework}' is not supported."

            # Execute Sandbox Child Verification
            sandbox_res = CompilationSandbox.validate_compilation(
                code=generated_code,
                framework=project.framework,
                project_id=str(project.id)
            )

            if sandbox_res["success"]:
                compiler_logger.info(
                    "Compilation completed successfully",
                    project_id=project_id_str,
                    task_id=self.request.id
                )
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
                compiler_logger.error(
                    "Compilation validation failed in sandbox",
                    project_id=project_id_str,
                    task_id=self.request.id,
                    errors=sandbox_res["compilation_errors"]
                )
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
            compiler_logger.error(
                "Internal compiler exception",
                project_id=project_id_str,
                task_id=self.request.id,
                error=str(compile_err)
            )
            # Fatal error inside generator/optimizer
            errs = [f"Internal compiler crash: {str(compile_err)}"]
            EventDispatcher.dispatch_validation_failed(proj_uuid, errs)
            return {"success": False, "errors": errs}

    except Exception as exc:
        compiler_logger.error(
            "Compilation task exception, retrying",
            project_id=project_id_str,
            task_id=self.request.id,
            error=str(exc)
        )
        # DB connection crashes or worker problems: retry task
        try:
            self.retry(exc=exc)
        except Exception as retry_exc:
            errs = [f"Task orchestration crash: {str(exc)}"]
            EventDispatcher.dispatch_validation_failed(proj_uuid, errs)
            return {"success": False, "errors": errs}
    finally:
        db.close()


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True
)
def async_dataset_verification_preflight(self, dataset_id_str: str, training_job_id_str: str):
    """Preflight task to verify dataset exists and is processed. If not, processes it.
    If ingestion fails, it updates the TrainingJob status to FAILED.
    """
    compiler_logger.info(
        "Dataset preflight verification started",
        dataset_id=dataset_id_str,
        training_job_id=training_job_id_str,
        task_id=self.request.id
    )
    db = SessionLocal()
    try:
        from app.models.dataset import Dataset
        from app.models.training_job import TrainingJob
        from app.tasks.dataset_tasks import async_process_dataset
        
        dataset_uuid = uuid.UUID(dataset_id_str)
        job_uuid = uuid.UUID(training_job_id_str)
        
        dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
        job = db.query(TrainingJob).filter(TrainingJob.id == job_uuid).first()
        
        if not dataset or not job:
            raise ValueError("Dataset or TrainingJob not found.")
            
        # Run dataset processing synchronously inside this task if not READY
        if dataset.status != "READY":
            res = async_process_dataset.run(dataset_id_str)
            if not res or not res.get("success"):
                raise ValueError(res.get("error", "Dataset processing failed during preflight."))
                
        # Double check status
        db.refresh(dataset)
        if dataset.status != "READY":
            raise ValueError("Dataset is not in READY state after processing.")
            
        compiler_logger.info(
            "Dataset preflight verification completed successfully",
            dataset_id=dataset_id_str,
            training_job_id=training_job_id_str,
            task_id=self.request.id
        )
        return {"success": True, "dataset_id": dataset_id_str}
        
    except Exception as e:
        compiler_logger.error(
            "Dataset preflight verification failed",
            dataset_id=dataset_id_str,
            training_job_id=training_job_id_str,
            task_id=self.request.id,
            error=str(e)
        )
        db.rollback()
        try:
            job_uuid = uuid.UUID(training_job_id_str)
            failed_job = db.query(TrainingJob).filter(TrainingJob.id == job_uuid).first()
            if failed_job:
                failed_job.status = "FAILED"
                failed_job.metrics_metadata = {"error": f"Dataset preflight failed: {str(e)}"}
                db.commit()
        except Exception as db_err:
            pass
        raise e
    finally:
        db.close()
