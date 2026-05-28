import os
import uuid
from sqlalchemy.orm import Session
from app.models.dataset import Dataset
from app.services.s3_service import S3Service
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
        description: str | None = None
    ) -> tuple[Dataset, str]:
        """Creates a Dataset database record in PENDING_UPLOAD status
        and generates the correct AWS S3 or Local fallback upload URL.
        """
        dataset_id = uuid.uuid4()
        
        # Determine the file path that the parser will inspect
        aws_key = settings.AWS_ACCESS_KEY_ID
        aws_secret = settings.AWS_SECRET_ACCESS_KEY
        bucket = settings.AWS_BUCKET_NAME

        if aws_key and aws_secret and bucket:
            # S3 Ingestion URI
            file_path = f"s3://{bucket}/datasets/{str(dataset_id)}/{filename}"
        else:
            # Local Storage Ingestion path
            workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            file_path = os.path.join(workspace_dir, "scratch", "storage", str(dataset_id), filename)

        new_dataset = Dataset(
            id=dataset_id,
            user_id=user_id,
            name=name.strip(),
            description=description.strip() if description else None,
            dataset_type=dataset_type.strip().upper(),
            status="PENDING_UPLOAD",
            file_path=file_path,
            num_records=0,
            schema_metadata=None
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
            raise PermissionError("You do not have permission to access this dataset.")
            
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
    def trigger_dataset_processing(db: Session, dataset_id: uuid.UUID, user_id: uuid.UUID) -> str:
        """Enqueues the async metadata extraction task for a dataset."""
        dataset = DatasetService.get_dataset(db, dataset_id, user_id=user_id)
        if not dataset:
            raise ValueError("Dataset not found.")
        
        # Enqueue Celery parsing task
        task = async_process_dataset.delay(str(dataset_id))
        return task.id
