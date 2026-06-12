import strawberry
from typing import List, Optional

@strawberry.type
class DatasetImageStatsType:
    image_count: int
    classes: List[str]
    class_counts: strawberry.scalars.JSON
    min_resolution: List[int]
    max_resolution: List[int]
    imbalance_ratio: float
    is_imbalanced: bool

@strawberry.type
class DatasetCSVStatsType:
    missing_values: strawberry.scalars.JSON
    outliers: strawberry.scalars.JSON
    correlations: strawberry.scalars.JSON

@strawberry.type
class DatasetTextStatsType:
    vocab_size: int
    total_tokens: int
    top_tokens: strawberry.scalars.JSON
    min_seq_len: int
    max_seq_len: int
    mean_seq_len: float

@strawberry.type
class DatasetAnalysisReportType:
    format: str
    row_count: int
    column_count: int
    image_stats: Optional[DatasetImageStatsType]
    csv_stats: Optional[DatasetCSVStatsType]
    text_stats: Optional[DatasetTextStatsType]
    recommendations: List[str]

@strawberry.type
class ExperimentAnalysisReportType:
    fit_type: str
    is_stable: bool
    loss_history: List[float]
    accuracy_history: List[float]
    val_loss_history: List[float]
    val_accuracy_history: List[float]
    recommendations: List[str]

@strawberry.type
class CostEstimateType:
    training_cost: float
    inference_cost_per_million: float
    gpu_hourly_cost: float
    storage_monthly_cost: float
    estimated_training_time_hours: float
    estimated_inference_latency_ms: float

@strawberry.type
class ExplainabilityReportType:
    shape_propagation: str
    attention_scaling: str
    vram_usage: str
    parameter_counts: str
    compiler_decisions: str
