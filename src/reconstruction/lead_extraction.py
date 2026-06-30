"""
Segments full 2D ECG pages into individual lead panels.
"""

from typing import Dict
import numpy as np
from ..utils.config import Config
from ..utils.logger import SetupLogger

logger = SetupLogger.get_logger(__name__)


class LeadExtractor:
    """
    Identifies and crops individual bounding boxes corresponding to distinct lead signals.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the lead extractor.

        Args:
            config: Global configuration object.
        """
        self.config = config
        
        # Standard layout configuration: 3 rows, 4 columns
        self.rows = 3
        self.cols = 4
        self.lead_names = [
            ["I", "aVR", "V1", "V4"],
            ["II", "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"]
        ]

    def extract_leads(self, binary_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Partitions the global mask image into separate lead coordinate masks.

        Args:
            binary_mask: The thresholded global binary segmentation mask.

        Returns:
            A dictionary mapping lead name strings to cropped grayscale arrays.
        """
        h, w = binary_mask.shape
        col_w = w // self.cols
        row_h = h // self.rows
        
        extracted: Dict[str, np.ndarray] = {}
        
        for col in range(self.cols):
            for row in range(self.rows):
                lead_name = self.lead_names[row][col]
                
                # Slicing bounds
                x_start = col * col_w
                y_start = row * row_h
                
                # Crop area (apply offsets to exclude labels and margins)
                crop_mask = binary_mask[
                    y_start + 45 : y_start + row_h - 10,
                    x_start + 10 : x_start + col_w - 10
                ]
                
                extracted[lead_name] = crop_mask
                
        logger.info(f"Successfully segmented mask into {len(extracted)} lead panels.")
        return extracted
