"""
Orchestrates the entire digitization flow from 2D mask to 1D signal.
"""

import os
from typing import Dict
import cv2
import numpy as np
import pandas as pd
from ..utils.config import Config
from ..utils.logger import SetupLogger
from .lead_extraction import LeadExtractor
from .signal_reconstruction import SignalReconstructor

logger = SetupLogger.get_logger(__name__)


class ECGDigitizer:
    """
    Coordinates the process of slicing masks into lead regions and converting
    each segmented panel into calibrated time-series dataframes.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the digitizer.

        Args:
            config: Global configuration object.
        """
        self.config = config
        self.extractor = LeadExtractor(self.config)
        self.reconstructor = SignalReconstructor(self.config)
        
        self.threshold: float = self.config.get("reconstruction.threshold", 0.5)
        self.output_dir: str = self.config.get(
            "paths.reconstructed_dir", "outputs/reconstructed_signals"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def digitize_mask(self, mask_img_path: str) -> pd.DataFrame:
        """
        Runs the full digitization process on a segmentation mask image.

        Args:
            mask_img_path: File path to the binary mask.

        Returns:
            A pandas DataFrame where each column is an extracted lead trace.
        """
        # Load grayscale mask
        mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask image: {mask_img_path}")

        # Binarize
        _, binary_mask = cv2.threshold(
            mask, int(self.threshold * 255), 255, cv2.THRESH_BINARY
        )

        # Slice into individual leads
        lead_masks = self.extractor.extract_leads(binary_mask)

        # Reconstruct 1D traces
        digitized_signals: Dict[str, np.ndarray] = {}
        for lead_name, lead_mask in lead_masks.items():
            digitized_signals[lead_name] = self.reconstructor.reconstruct_trace(lead_mask)

        # Combine into DataFrame
        df_signals = pd.DataFrame(digitized_signals)

        # Save to output folder
        base_name = os.path.splitext(os.path.basename(mask_img_path))[0]
        save_path = os.path.join(self.output_dir, f"{base_name}_digitized.csv")
        df_signals.to_csv(save_path, index=False)
        
        logger.info(f"Digitization completed. Output saved to {save_path}")
        return df_signals
