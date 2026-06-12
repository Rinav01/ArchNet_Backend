import uuid
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.workflow import Workflow
from app.models.workflow_run import WorkflowRun
from app.models.project import Project
from app.models.dataset import Dataset
from app.models.training_run import TrainingRun
from app.models.model_artifact import ModelArtifact
from app.services.deployment_service import DeploymentService
from app.services.experiment_analysis_service import ExperimentAnalysisService

logger = logging.getLogger("mlbuilder.workflow_executor")

class WorkflowExecutor:
    @staticmethod
    def execute_run(db: Session, run_id: uuid.UUID) -> dict:
        """
        Executes a WorkflowRun based on its action type.
        Updates status to RUNNING, COMPLETED, or FAILED and captures logs.
        """
        run = db.query(WorkflowRun).filter(WorkflowRun.id == run_id).first()
        if not run:
            raise ValueError("WorkflowRun not found.")

        run.status = "RUNNING"
        db.commit()

        logs = [f"Starting execution for workflow: '{run.workflow.name}'"]
        logs.append(f"Action: {run.workflow.action_type}")
        logs.append(f"Trigger Event: {run.trigger_event}")
        logs.append(f"Trigger Resource: {run.triggered_by_resource_id}")

        try:
            workflow = run.workflow
            project_id = workflow.project_id
            config = workflow.config or {}

            if workflow.action_type == "ANALYZE_DATASET":
                # Find dataset to analyze
                dataset_id = None
                if run.triggered_by_resource_id:
                    try:
                        dataset_id = uuid.UUID(run.triggered_by_resource_id)
                    except ValueError:
                        pass
                
                if not dataset_id and project_id:
                    # Fallback to the project's dataset
                    ds = db.query(Dataset).filter(Dataset.project_id == project_id).first()
                    if ds:
                        dataset_id = ds.id

                if not dataset_id:
                    raise ValueError("Dataset not found for analysis.")

                logs.append(f"Triggering asynchronous dataset analysis for ID: {dataset_id}")
                from app.tasks.dataset_tasks import async_process_dataset
                async_process_dataset.delay(str(dataset_id))
                logs.append("Dataset analysis Celery task enqueued successfully.")

            elif workflow.action_type == "COMPARE_RUNS":
                if not project_id:
                    raise ValueError("Workflow must be associated with a project to compare runs.")
                
                # Fetch recent runs
                runs = db.query(TrainingRun).filter(TrainingRun.project_id == project_id).order_by(TrainingRun.created_at.desc()).limit(5).all()
                if not runs:
                    logs.append("No training runs found to compare.")
                else:
                    logs.append(f"Comparing {len(runs)} recent training runs:")
                    for r in runs:
                        logs.append(f"- Run ID: {r.id}, Loss: {r.loss:.4f}, Accuracy: {r.accuracy:.4f}, Created: {r.created_at}")
                    
                    # Run detailed curve analysis for the latest run
                    latest_run = runs[0]
                    analysis = ExperimentAnalysisService.analyze_experiment_run(db, latest_run.id)
                    logs.append(f"Latest run fit: {analysis['fit_type']}")
                    logs.append(f"Stability: {'Stable' if analysis['is_stable'] else 'Unstable'}")
                    logs.append("Recommendations:")
                    for rec in analysis["recommendations"]:
                        logs.append(f"  * {rec}")

            elif workflow.action_type == "DEPLOY_MODEL":
                # Find the latest model artifact for the project
                if not project_id:
                    raise ValueError("Project ID is missing.")

                artifact = db.query(ModelArtifact).filter(ModelArtifact.project_id == project_id).order_by(ModelArtifact.created_at.desc()).first()
                if not artifact:
                    raise ValueError("No ModelArtifact found to deploy.")

                target = config.get("target", "FastAPI")
                logs.append(f"Auto-deploying ModelArtifact {artifact.id} to target: {target}")
                
                deployment = DeploymentService.deploy_artifact(db, artifact.id, target)
                logs.append(f"Deployment created successfully with ID: {deployment.id}")
                logs.append(f"Endpoint URL: {deployment.endpoint_url}")

            elif workflow.action_type == "SEND_ALERTS":
                channel = config.get("channel", "email")
                recipient = config.get("recipient", "admin@mlbuilder.com")
                message = config.get("message", "MLBuilder Automation Notification")
                
                logs.append(f"Simulating Alert Notification via {channel.upper()}:")
                logs.append(f"Recipient: {recipient}")
                logs.append(f"Message: {message}")
                logs.append("Alert dispatched successfully.")

            else:
                raise ValueError(f"Unknown action type: {workflow.action_type}")

            run.status = "COMPLETED"
            logs.append("Workflow execution finished successfully.")
        except Exception as e:
            run.status = "FAILED"
            logs.append(f"Execution failed with error: {str(e)}")
            logger.error(f"Workflow execution failed: {e}", exc_info=True)

        run.execution_logs = "\n".join(logs)
        run.updated_at = datetime.utcnow()
        db.commit()
        return {"status": run.status, "logs": run.execution_logs}
