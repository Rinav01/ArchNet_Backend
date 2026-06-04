import os
import sys
import uuid
import time
import zipfile
import shutil
import logging
import importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.models.project import Project
from app.models.node import Node
from app.models.edge import Edge
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.services.event_dispatcher import EventDispatcher
from app.services.validation_service import ValidationService
from app.services.shape_inference_service import ShapeInferenceService
from app.ir.ir_graph import IRGraph
from app.services.graph_engine import GraphOptimizer
from app.codegen.pytorch.generator import PyTorchCompiler
from app.config.logging import training_logger

logger = training_logger

class TrainingService:
    @staticmethod
    def get_local_dataset_path(dataset: Dataset) -> tuple[str, bool]:
        """
        Resolves the dataset's storage path. If remote (S3/GCS), downloads it
        to a local scratch directory. Returns (local_path, is_temporary).
        """
        file_path = dataset.storage_path
        if not file_path:
            raise ValueError("Dataset storage path is empty.")

        # Resolve paths starting with s3:// or gs://
        if file_path.startswith("s3://"):
            import boto3
            from urllib.parse import urlparse
            parsed = urlparse(file_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            scratch_temp = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "scratch", "temp"))
            os.makedirs(scratch_temp, exist_ok=True)
            local_path = os.path.join(scratch_temp, f"s3_{uuid.uuid4().hex}_{os.path.basename(file_path)}")

            logger.info(f"Downloading dataset from S3: {file_path} -> {local_path}")
            s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            s3.download_file(bucket, key, local_path)
            return local_path, True

        elif file_path.startswith("gs://"):
            from google.cloud import storage
            from google.oauth2 import service_account
            from urllib.parse import urlparse
            import json

            parsed = urlparse(file_path)
            bucket_name = parsed.netloc
            blob_name = parsed.path.lstrip("/")

            scratch_temp = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "scratch", "temp"))
            os.makedirs(scratch_temp, exist_ok=True)
            local_path = os.path.join(scratch_temp, f"gcs_{uuid.uuid4().hex}_{os.path.basename(file_path)}")

            logger.info(f"Downloading dataset from GCS: {file_path} -> {local_path}")
            if settings.GCP_CREDENTIALS_JSON:
                creds_dict = json.loads(settings.GCP_CREDENTIALS_JSON)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                storage_client = storage.Client(credentials=credentials, project=settings.GCP_PROJECT_ID)
            else:
                storage_client = storage.Client()

            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            return local_path, True

        else:
            # Local Storage Path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Local dataset file not found: {file_path}")
            return file_path, False

    @staticmethod
    def load_dataset_and_create_dataloader(
        local_path: str,
        dataset_type: str,
        batch_size: int = 32,
        input_shape: list | None = None
    ) -> tuple[DataLoader, int, int, str | None]:
        """
        Loads dataset based on dataset_type. Returns:
        (dataloader, input_dim, output_dim, temp_directory_path)
        """
        dtype = dataset_type.lower().strip()
        temp_dir = None

        if "csv" in dtype:
            df = pd.read_csv(local_path)
            # Encode non-numeric/object columns
            for col in df.columns:
                if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
                    df[col] = df[col].astype('category').cat.codes
            df = df.fillna(0.0)

            # Features are all but the last column, labels is the last column
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(np.float32)

            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            input_dim = X.shape[1] if len(X.shape) > 1 else 1
            unique_ys = np.unique(y)
            if np.all(y == y.astype(int)) and len(unique_ys) < 100:
                output_dim = int(unique_ys.max() + 1)
            else:
                output_dim = 1

            return dataloader, input_dim, output_dim, None

        elif "image_zip" in dtype or "zip" in dtype:
            from torchvision import datasets, transforms
            temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "scratch", "temp", f"zip_extract_{uuid.uuid4().hex}"))
            os.makedirs(temp_dir, exist_ok=True)

            logger.info(f"Extracting dataset ZIP to temp path: {temp_dir}")
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Infer image size from model config
            img_h, img_w = 224, 224
            if input_shape and len(input_shape) >= 4:
                img_h = input_shape[2] or 224
                img_w = input_shape[3] or 224

            transform = transforms.Compose([
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Resolve actual data path (in case of single parent folder nesting inside ZIP)
            target_dir = temp_dir
            subdirs = [os.path.join(temp_dir, d) for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            if len(subdirs) == 1 and not any(os.path.isfile(os.path.join(temp_dir, f)) for f in os.listdir(temp_dir)):
                target_dir = subdirs[0]

            img_dataset = datasets.ImageFolder(target_dir, transform=transform)
            dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True)

            return dataloader, img_h * img_w * 3, len(img_dataset.classes), temp_dir

        elif "tensor" in dtype or "numpy" in dtype or "npy" in dtype:
            if local_path.endswith('.npz'):
                data = np.load(local_path)
                X_key = next((k for k in data.keys() if k.lower() in ('x', 'features', 'inputs', 'data')), None)
                y_key = next((k for k in data.keys() if k.lower() in ('y', 'labels', 'targets', 'outputs')), None)

                if X_key and y_key:
                    X = data[X_key].astype(np.float32)
                    y = data[y_key].astype(np.float32)
                else:
                    keys = list(data.keys())
                    if len(keys) >= 2:
                        X = data[keys[0]].astype(np.float32)
                        y = data[keys[1]].astype(np.float32)
                    else:
                        raise ValueError("NPZ file does not contain distinct features and label arrays.")
            else:
                arr = np.load(local_path)
                if arr.ndim >= 2:
                    X = arr[..., :-1].astype(np.float32)
                    y = arr[..., -1].astype(np.float32)
                else:
                    X = arr.astype(np.float32)
                    y = np.zeros(len(X), dtype=np.float32)

            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.unsqueeze(1)

            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            input_dim = X.shape[1] if X.ndim > 1 else 1
            unique_ys = np.unique(y)
            if np.all(y == y.astype(int)) and len(unique_ys) < 100:
                output_dim = int(unique_ys.max() + 1)
            else:
                output_dim = 1

            return dataloader, input_dim, output_dim, None

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    @staticmethod
    def instantiate_model(generated_code: str) -> nn.Module:
        """
        Executes the dynamically generated PyTorch compiler code and instantiates the MLBuilderModel.
        """
        namespace = {}
        # Execute generated script to declare class
        exec(generated_code, namespace)

        cls_name = next((k for k in ("MLBuilderModel", "GeneratedModel") if k in namespace), None)
        if not cls_name:
            raise ValueError("Compiled model class (MLBuilderModel or GeneratedModel) was not found in executed namespace.")

        model = namespace[cls_name]()
        return model

    @classmethod
    def run_pipeline(
        cls,
        db: Session,
        project_id: uuid.UUID,
        dataset_id: uuid.UUID | None,
        epochs: int,
        training_job_id: uuid.UUID
    ) -> dict:
        """
        Completes the training pipeline: Project compiling -> Load dataset -> Instantiation -> Epoch training/validation loops -> Metrics saving.
        """
        logger.info(f"Starting training pipeline run for project: {project_id}, job: {training_job_id}")
        start_time = time.time()
        
        # Load Training Job
        job = db.query(TrainingJob).filter(TrainingJob.id == training_job_id).first()
        if not job:
            raise ValueError(f"TrainingJob {training_job_id} not found.")

        # Update job to RUNNING
        job.status = "RUNNING"
        job.current_epoch = 0
        job.loss_history = []
        job.accuracy_history = []
        db.commit()

        # Publish TrainingStarted WS event
        EventDispatcher.get_redis().publish(
            "mlbuilder:project:training",
            f'{{"type": "TrainingStarted", "training_job_id": "{str(training_job_id)}", "epochs": {epochs}}}'
        )

        local_path = None
        is_temp_file = False
        temp_dir = None

        try:
            # 1. Compile project graph to PyTorch code
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError("Project not found.")

            nodes = db.query(Node).filter(Node.project_id == project_id).all()
            edges = db.query(Edge).filter(Edge.project_id == project_id).all()

            # Topological sort & shape inference validation
            sorted_nodes = ValidationService.validate_graph(nodes, edges)
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
            db.commit()

            # Get target input shape
            input_shape = None
            for node in sorted_nodes:
                if node.type.lower() == "input":
                    input_shape = node.output_shape
                    break

            # Simplify & Compile
            ir_graph = IRGraph.from_db(project, sorted_nodes, edges)
            GraphOptimizer.simplify_graph(ir_graph)
            compiler = PyTorchCompiler()
            generated_code = compiler.compile(ir_graph)

            # 2. Load Dataset & Create DataLoader
            dataloader = None
            dataset_type = "CSV"
            if dataset_id:
                dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
                if not dataset:
                    raise ValueError("Dataset not found.")
                dataset_type = dataset.dataset_type
                local_path, is_temp_file = cls.get_local_dataset_path(dataset)
                dataloader, input_dim, output_dim, temp_dir = cls.load_dataset_and_create_dataloader(
                    local_path,
                    dataset_type,
                    batch_size=32,
                    input_shape=input_shape
                )
            else:
                # Default mock TensorDataset fallback if no dataset is linked
                logger.info("No dataset linked. Creating standard dummy DataLoader for model testing.")
                test_input_dims = [128, 3, 224, 224] if input_shape and len(input_shape) >= 4 else [128, 10]
                if input_shape and len(input_shape) > 1:
                    test_input_dims = [128] + [dim if dim is not None else 1 for dim in input_shape[1:]]

                X_dummy = torch.randn(*test_input_dims)
                y_dummy = torch.randn(128, 1)
                dataset = TensorDataset(X_dummy, y_dummy)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                dataset_type = "MOCK_TENSOR"

            # 3. Dynamic Model Instantiation
            model = cls.instantiate_model(generated_code)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            logger.info(f"Loaded model successfully onto device: {device}")

            # Define loss and optimizer
            is_classification = "zip" in dataset_type.lower()
            if is_classification:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            loss_history = []
            accuracy_history = []

            # 4. Training/Validation Loop
            for epoch in range(1, epochs + 1):
                # Verify cancellation check
                db.refresh(job)
                if job.status == "CANCELLED":
                    logger.info(f"Training job {training_job_id} cancelled by user.")
                    return {"success": False, "error": "Job cancelled by user."}

                model.train()
                epoch_loss = 0.0
                correct = 0
                total = 0

                for data, targets in dataloader:
                    data, targets = data.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(data)

                    # Dynamic target handling to avoid shape mismatches
                    if not is_classification:
                        if outputs.shape != targets.shape:
                            if outputs.ndim == targets.ndim:
                                targets = targets.expand_as(outputs)
                            else:
                                targets = targets.view_as(outputs)
                        loss = criterion(outputs, targets)
                    else:
                        loss = criterion(outputs, targets.long())

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * data.size(0)

                    # Calculate batch accuracy
                    if is_classification:
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                    else:
                        # Regression relative accuracy (R2-style proxy metrics)
                        total += targets.size(0)
                        correct += max(0.0, 1.0 - loss.item()) * targets.size(0)

                epoch_loss /= len(dataloader.dataset)
                epoch_acc = correct / max(1, total)

                loss_history.append(round(epoch_loss, 4))
                accuracy_history.append(round(epoch_acc, 4))

                # Update database epoch telemetry
                job.current_epoch = epoch
                job.loss_history = loss_history
                job.accuracy_history = accuracy_history
                db.commit()

                # Publish telemetry to websocket channels
                EventDispatcher.get_redis().publish(
                    "mlbuilder:project:training",
                    f'{{"type": "TrainingEpochProgress", "training_job_id": "{str(training_job_id)}", "current_epoch": {epoch}, "loss": {epoch_loss:.4f}, "accuracy": {epoch_acc:.4f}}}'
                )

                # Simulate small hardware sleep representing real compute intervals
                time.sleep(0.1)

            # 5. Completed successfully! Save final metrics metadata
            duration = time.time() - start_time
            job.status = "COMPLETED"
            job.metrics_metadata = {
                "device": str(device),
                "peak_memory_used_mb": 0 if str(device) == "cpu" else torch.cuda.max_memory_allocated(device) // (1024 * 1024),
                "training_duration_seconds": round(duration, 2),
                "final_loss": loss_history[-1],
                "final_accuracy": accuracy_history[-1],
                "logs": "Training pipeline runs finished successfully. Dynamic model loaded and computed correctly."
            }
            db.commit()

            # Save Experiment Tracking Run (Backend Module 5)
            from app.models.training_run import TrainingRun
            run = TrainingRun(
                id=uuid.uuid4(),
                project_id=project_id,
                training_job_id=training_job_id,
                accuracy=accuracy_history[-1],
                loss=loss_history[-1],
                metrics_json=job.metrics_metadata,
                config_json={
                    "epochs": epochs,
                    "dataset_id": str(dataset_id) if dataset_id else None,
                    "dataset_type": dataset_type
                }
            )
            db.add(run)
            db.commit()

            # Publish completed event
            EventDispatcher.get_redis().publish(
                "mlbuilder:project:training",
                f'{{"type": "TrainingCompleted", "training_job_id": "{str(training_job_id)}", "status": "COMPLETED", "accuracy": {accuracy_history[-1]}}}'
            )

            return {
                "success": True,
                "training_job_id": str(training_job_id),
                "final_loss": loss_history[-1],
                "final_accuracy": accuracy_history[-1]
            }

        except Exception as e:
            logger.error(f"Training pipeline crashed on model: {project_id}. Error: {e}", exc_info=True)
            db.rollback()
            
            # Update database status to FAILED
            failed_job = db.query(TrainingJob).filter(TrainingJob.id == training_job_id).first()
            if failed_job and failed_job.status != "CANCELLED":
                failed_job.status = "FAILED"
                failed_job.metrics_metadata = {
                    "error": str(e),
                    "logs": f"Training failed due to crash: {str(e)}"
                }
                db.commit()

            # Publish failure event
            EventDispatcher.get_redis().publish(
                "mlbuilder:project:training",
                f'{{"type": "TrainingFailed", "training_job_id": "{str(training_job_id)}", "status": "FAILED", "error": "{str(e)}"}}'
            )
            return {"success": False, "error": str(e)}

        finally:
            # 6. Cleanup local temporary downloaded files or extracted ZIP files
            if is_temp_file and local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    logger.info(f"Cleaned up temp dataset file: {local_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to clean up temp dataset file {local_path}: {cleanup_err}")

            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temp extracted directory: {temp_dir}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to remove temp extracted directory {temp_dir}: {cleanup_err}")
