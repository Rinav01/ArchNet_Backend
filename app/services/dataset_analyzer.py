import os
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List

class DatasetAnalyzer:
    @staticmethod
    def analyze_csv(filepath: str) -> Dict[str, Any]:
        """Validates and analyzes a CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        # Read sample
        df_sample = pd.read_csv(filepath, nrows=100)
        
        columns = []
        for col_name in df_sample.columns:
            dtype = df_sample[col_name].dtype
            if np.issubdtype(dtype, np.number):
                col_type = "numeric"
            elif np.issubdtype(dtype, np.datetime64):
                col_type = "datetime"
            else:
                col_type = "categorical"
            columns.append({"name": col_name, "type": col_type})

        # Calculate exact row count
        row_count = sum(1 for _ in open(filepath, "r", encoding="utf-8", errors="ignore")) - 1
        if row_count < 0:
            row_count = 0
            
        column_count = len(df_sample.columns)
        
        # Get preview of first 5 rows
        preview_data = df_sample.head(5).to_dict(orient="records")

        return {
            "row_count": row_count,
            "column_count": column_count,
            "metadata_json": {
                "format": "CSV",
                "columns": columns,
                "preview_data": preview_data
            }
        }

    @staticmethod
    def analyze_image_zip(filepath: str) -> Dict[str, Any]:
        """Validates and analyzes a ZIP image dataset."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ZIP file not found: {filepath}")

        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        image_count = 0
        formats = set()
        channels = set()
        classes = set()
        
        min_w, min_h = float("inf"), float("inf")
        max_w, max_h = 0, 0

        with zipfile.ZipFile(filepath, "r") as zip_ref:
            # Read files list
            file_list = []
            for name in zip_ref.namelist():
                ext = os.path.splitext(name.lower())[1]
                if ext in valid_extensions:
                    file_list.append(name)
                    
                    # Extract class folders: e.g. "cats/cat_1.png" -> folder name "cats"
                    parts = name.split("/")
                    if len(parts) > 1 and parts[-2]:
                        classes.add(parts[-2])

            image_count = len(file_list)

            # Sample up to 10 images for channels/resolutions
            sample_files = file_list[:10]
            for file_name in sample_files:
                with zip_ref.open(file_name) as file:
                    try:
                        with Image.open(file) as img:
                            w, h = img.size
                            min_w = min(min_w, w)
                            min_h = min(min_h, h)
                            max_w = max(max_w, w)
                            max_h = max(max_h, h)
                            formats.add(img.format)
                            channels.add(img.mode)
                    except Exception:
                        pass

        if image_count == 0:
            min_w, min_h, max_w, max_h = 0, 0, 0, 0

        metadata = {
            "format": "IMAGE_ZIP",
            "image_count": image_count,
            "classes": sorted(list(classes)),
            "min_resolution": [int(min_w) if min_w != float("inf") else 0, int(min_h) if min_h != float("inf") else 0],
            "max_resolution": [int(max_w), int(max_h)],
            "formats": list(formats),
            "channels": list(channels)
        }
        
        return {
            "row_count": image_count,
            "column_count": len(classes),
            "metadata_json": metadata
        }
        
    @staticmethod
    def analyze_dataset(filepath: str, dataset_type: str) -> Dict[str, Any]:
        """Analyzes a dataset based on its type."""
        dtype = dataset_type.lower().strip()
        if "csv" in dtype:
            return DatasetAnalyzer.analyze_csv(filepath)
        elif "image_zip" in dtype or "zip" in dtype:
            return DatasetAnalyzer.analyze_image_zip(filepath)
        elif "tensor" in dtype or "numpy" in dtype or "npy" in dtype:
            import numpy as np
            arr = np.load(filepath, mmap_mode="r")
            shape = list(arr.shape)
            num_records = shape[0] if shape else 0
            
            return {
                "row_count": num_records,
                "column_count": shape[1] if len(shape) > 1 else 1,
                "metadata_json": {
                    "format": "NUMPY_TENSOR",
                    "shape": shape,
                    "dtype": str(arr.dtype),
                    "rank": arr.ndim
                }
            }
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_type}")
