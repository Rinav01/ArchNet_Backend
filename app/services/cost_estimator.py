import uuid
from sqlalchemy.orm import Session
from app.models.project import Project
from app.models.node import Node
from app.models.edge import Edge
from app.models.dataset import Dataset
from app.services.memory_estimator import MemoryEstimator
from app.services.validation_service import ValidationService
from app.services.shape_inference_service import ShapeInferenceService

class CostEstimator:
    # Hourly pricing rates for different GPU/CPU compute targets
    GPU_RATES = {
        "T4": 0.35,      # $0.35 per hour
        "V100": 2.48,    # $2.48 per hour
        "A100": 3.67,    # $3.67 per hour
        "H100": 4.76,    # $4.76 per hour
        "CPU": 0.05      # $0.05 per hour
    }

    # Realistic operational FLOPs throughput per second taking into account framework overheads (MFU ~30%)
    GPU_FLOPS_PER_SEC = {
        "T4": 1e11,      # 100 GFLOPs/sec realistic utilization
        "V100": 5e11,     # 500 GFLOPs/sec
        "A100": 2e12,     # 2 TFLOPs/sec
        "H100": 1e13,     # 10 TFLOPs/sec
        "CPU": 5e9        # 5 GFLOPs/sec
    }

    @staticmethod
    def estimate_costs_for_project(
        db: Session,
        project_id: uuid.UUID,
        dataset_id: uuid.UUID | None = None,
        epochs: int = 10,
        gpu_type: str = "T4"
    ) -> dict:
        """
        Calculates expected training cost, inference cost per million runs, monthly GCS/S3 storage, 
        and hardware performance benchmarks using compiled model parameters.
        """
        # Fetch project components
        nodes = db.query(Node).filter(Node.project_id == project_id).all()
        edges = db.query(Edge).filter(Edge.project_id == project_id).all()

        # Run shape inference and topological sorting to ensure up-to-date node parameters
        try:
            sorted_nodes = ValidationService.validate_graph(nodes, edges)
            ShapeInferenceService.run_shape_inference(sorted_nodes, edges)
            db.commit()
            nodes = db.query(Node).filter(Node.project_id == project_id).all()
        except Exception:
            pass

        # Estimate parameters count, FLOPs complexity and VRAM footprint
        metrics = MemoryEstimator.estimate_project_metrics(nodes)
        total_flops = metrics["total_flops"]
        parameter_count = metrics["total_parameter_count"]
        vram_mb = metrics["estimated_gpu_memory_mb"]

        # Resolve dataset sample size
        dataset_records = 10000
        if dataset_id:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                dataset_records = dataset.num_records or 10000

        # Validate GPU/CPU selection
        gpu_type_clean = gpu_type.upper() if gpu_type else "T4"
        if gpu_type_clean not in CostEstimator.GPU_RATES:
            gpu_type_clean = "T4"

        hourly_rate = CostEstimator.GPU_RATES[gpu_type_clean]
        flops_per_sec = CostEstimator.GPU_FLOPS_PER_SEC[gpu_type_clean]

        # 1. Training Cost calculation
        # Backpropagation forward/backward passes take roughly 3x forward FLOPs complexity
        total_training_flops = 3.0 * total_flops * dataset_records * epochs
        estimated_training_time_sec = total_training_flops / flops_per_sec
        estimated_training_time_hours = max(0.001, estimated_training_time_sec / 3600.0)
        training_cost = estimated_training_time_hours * hourly_rate

        # 2. Inference Cost calculation
        # Forward pass runs sequentially
        estimated_inference_time_sec = total_flops / flops_per_sec
        estimated_inference_latency_ms = estimated_inference_time_sec * 1000.0
        inference_cost_per_million = (estimated_inference_time_sec * 1000000.0 / 3600.0) * hourly_rate

        # 3. Storage Cost calculation
        # Monthly standard storage charges ($0.023/GB) assuming avg record size is 1KB
        estimated_size_gb = (dataset_records * 1024.0) / 1e9
        storage_monthly_cost = estimated_size_gb * 0.023

        return {
            "training_cost": round(training_cost, 4),
            "inference_cost_per_million": round(inference_cost_per_million, 4),
            "gpu_hourly_cost": round(hourly_rate, 4),
            "storage_monthly_cost": round(storage_monthly_cost, 6),
            "estimated_training_time_hours": round(estimated_training_time_hours, 4),
            "estimated_inference_latency_ms": round(estimated_inference_latency_ms, 4)
        }
