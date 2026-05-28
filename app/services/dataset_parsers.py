import os
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any

class DatasetParserService:
    @staticmethod
    def parse_csv_metadata(filepath: str) -> Tuple[int, Dict[str, Any]]:
        """Parses a CSV file, extracting the row count and mapping the columns to datatypes."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        # Read only a chunk or headers first for safety, then load size
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

        # Calculate exact row count (fast on modern pandas/OS)
        row_count = sum(1 for _ in open(filepath, "r", encoding="utf-8", errors="ignore")) - 1
        if row_count < 0:
            row_count = 0
            
        metadata = {
            "columns": columns,
            "format": "CSV"
        }
        return row_count, metadata

    @staticmethod
    def parse_image_zip_metadata(filepath: str) -> Tuple[int, Dict[str, Any]]:
        """Parses a ZIP archive containing images, extracting total image count,
        color channels, formats, and resolution ranges.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Zip file not found: {filepath}")

        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        image_count = 0
        formats = set()
        channels = set()
        
        min_w, min_h = float("inf"), float("inf")
        max_w, max_h = 0, 0

        with zipfile.ZipFile(filepath, "r") as zip_ref:
            # Filter file list
            file_list = [f for f in zip_ref.namelist() if os.path.splitext(f.lower())[1] in valid_extensions]
            image_count = len(file_list)

            # Sample up to 10 images to determine resolutions and channels
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
                        # Skip corrupted files gracefully
                        pass

        # Handle empty zip case safely
        if image_count == 0:
            min_w, min_h, max_w, max_h = 0, 0, 0, 0

        metadata = {
            "image_count": image_count,
            "min_resolution": [int(min_w) if min_w != float("inf") else 0, int(min_h) if min_h != float("inf") else 0],
            "max_resolution": [int(max_w), int(max_h)],
            "formats": list(formats),
            "channels": list(channels)
        }
        return image_count, metadata

    @staticmethod
    def parse_tensor_metadata(filepath: str) -> Tuple[int, Dict[str, Any]]:
        """Parses a NumPy .npy tensor file using memory mapping, extracting shapes and array ranks."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tensor file not found: {filepath}")

        # Memory mapping allows reading headers without consuming RAM
        arr = np.load(filepath, mmap_mode="r")
        shape = list(arr.shape)
        num_records = shape[0] if shape else 0
        
        metadata = {
            "shape": shape,
            "dtype": str(arr.dtype),
            "rank": arr.ndim
        }
        return num_records, metadata
