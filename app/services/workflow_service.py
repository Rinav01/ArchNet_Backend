import uuid
from datetime import datetime
from typing import List
from sqlalchemy.orm import Session
from app.models.workflow import Workflow
from app.models.workflow_run import WorkflowRun
from app.services.workflow_executor import WorkflowExecutor

class WorkflowService:
    @staticmethod
    def create_workflow(
        db: Session,
        project_id: uuid.UUID | None,
        name: str,
        trigger_event: str,
        action_type: str,
        config: dict = None
    ) -> Workflow:
        """Creates a new automated workflow."""
        workflow = Workflow(
            id=uuid.uuid4(),
            project_id=project_id,
            name=name.strip(),
            trigger_event=trigger_event.strip(),
            action_type=action_type.strip(),
            config=config or {},
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(workflow)
        db.commit()
        db.refresh(workflow)
        return workflow

    @staticmethod
    def get_workflow(db: Session, workflow_id: uuid.UUID) -> Workflow | None:
        """Retrieves a workflow by its ID."""
        return db.query(Workflow).filter(Workflow.id == workflow_id).first()

    @staticmethod
    def list_workflows(db: Session, project_id: uuid.UUID | None = None) -> List[Workflow]:
        """Lists active workflows for a project (or global if project_id is None)."""
        query = db.query(Workflow)
        if project_id:
            query = query.filter(Workflow.project_id == project_id)
        return query.all()

    @staticmethod
    def delete_workflow(db: Session, workflow_id: uuid.UUID) -> bool:
        """Deletes a workflow."""
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if not workflow:
            return False
        db.delete(workflow)
        db.commit()
        return True

    @staticmethod
    def trigger_workflows_for_event(
        db: Session,
        event_type: str,
        resource_id: uuid.UUID | str,
        project_id: uuid.UUID | None = None
    ) -> List[WorkflowRun]:
        """
        Scans and executes all active workflows registered for a specific trigger event.
        """
        # Find active workflows that match the trigger event type
        query = db.query(Workflow).filter(
            Workflow.trigger_event == event_type,
            Workflow.is_active == True
        )
        if project_id:
            query = query.filter(
                (Workflow.project_id == project_id) | (Workflow.project_id == None)
            )
        workflows = query.all()

        runs = []
        for workflow in workflows:
            run = WorkflowRun(
                id=uuid.uuid4(),
                workflow_id=workflow.id,
                status="PENDING",
                trigger_event=event_type,
                triggered_by_resource_id=str(resource_id),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(run)
            db.commit()
            db.refresh(run)

            # Execute the workflow run immediately (synchronous for reliability in this handler)
            WorkflowExecutor.execute_run(db, run.id)
            db.refresh(run)
            runs.append(run)

        return runs
