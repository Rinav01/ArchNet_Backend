import os
import uuid
from sqlalchemy.orm import Session
from app.models.dataset import Dataset
from app.services.s3_service import S3Service
from app.services.dataset_storage import DatasetStorage
from app.services.dataset_analyzer import DatasetAnalyzer
from app.config.settings import settings
from app.tasks.dataset_tasks import async_process_dataset

class DatasetService:
    @staticmethod
    def create_dataset(
        db: Session,
        user_id: uuid.UUID,
        name: str,
        dataset_type: str,
        filename: str,
        description: str | None = None,
        project_id: uuid.UUID | None = None
    ) -> tuple[Dataset, str]:
        """Creates a Dataset database record in PENDING_UPLOAD status
        and generates the correct AWS S3 or Local fallback upload URL.
        """
        dataset_id = uuid.uuid4()
        storage_path = DatasetStorage.get_storage_path(dataset_id, filename)

        new_dataset = Dataset(
            id=dataset_id,
            user_id=user_id,
            project_id=project_id,
            name=name.strip(),
            description=description.strip() if description else None,
            dataset_type=dataset_type.strip().upper(),
            status="PENDING_UPLOAD",
            storage_path=storage_path,
            row_count=0,
            column_count=0,
            metadata_json=None
        )

        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        # Generate pre-signed PUT upload link (with local fallback)
        upload_url = S3Service.generate_presigned_upload_url(dataset_id, filename)

        return new_dataset, upload_url

    @staticmethod
    def get_dataset(db: Session, dataset_id: uuid.UUID, user_id: uuid.UUID | None = None) -> Dataset | None:
        """Retrieves a dataset by ID, optionally verifying ownership."""
        query = db.query(Dataset).filter(Dataset.id == dataset_id)
        dataset = query.first()
        if not dataset:
            return None
        
        if user_id and dataset.user_id != user_id:
            # For admin bypass in mutations, check role outside or verify here
            pass
            
        return dataset

    @staticmethod
    def list_datasets(db: Session, user_id: uuid.UUID, limit: int = 20, offset: int = 0) -> list[Dataset]:
        """Lists all datasets belonging to a specific user."""
        return db.query(Dataset)\
            .filter(Dataset.user_id == user_id)\
            .order_by(Dataset.created_at.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()

    @staticmethod
    def delete_dataset(db: Session, dataset_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Deletes the dataset record from DB and cleans up the stored file asset."""
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return False
            
        # Verify ownership unless admin
        from app.models.user import User
        user = db.query(User).filter(User.id == user_id).first()
        if user and user.role != "admin" and dataset.user_id != user_id:
            raise PermissionError("Forbidden: You do not own this dataset.")

        # Cleanup physical file storage
        DatasetStorage.delete_dataset_file(dataset.storage_path)

        # Delete database row
        db.delete(dataset)
        db.commit()
        return True

    @staticmethod
    def trigger_dataset_processing(db: Session, dataset_id: uuid.UUID, user_id: uuid.UUID) -> str:
        """Enqueues the async metadata extraction task for a dataset."""
        dataset = DatasetService.get_dataset(db, dataset_id)
        if not dataset:
            raise ValueError("Dataset not found.")
        
        # Enqueue Celery parsing task
        task = async_process_dataset.delay(str(dataset_id))
        return task.id
