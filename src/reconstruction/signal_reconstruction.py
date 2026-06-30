"""
Transforms 2D pixel coordinate grids into 1D time-series values.
"""

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from ..utils.config import Config
from ..utils.logger import SetupLogger

logger = SetupLogger.get_logger(__name__)


class SignalReconstructor:
    """
    Decodes active pixel points on sub-masks and maps grid metrics back to time/voltage.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the signal reconstructor.

        Args:
            config: Global configuration object.
        """
        self.config = config
        self.smoothing_window: int = self.config.get("reconstruction.smoothing_window", 5)

    def reconstruct_trace(self, lead_mask: np.ndarray) -> np.ndarray:
        """
        Converts a binary sub-mask of a single lead back into a 1D trace.

        Args:
            lead_mask: Grayscale sub-image mask (binary representation).

        Returns:
            A 1D numpy array representing the digitized signal trace.
        """
        h, w = lead_mask.shape
        y_values = []
        x_indices = []

        # Extract active pixel centroids column-by-column
        for x in range(w):
            active_y = np.where(lead_mask[:, x] > 0)[0]
            if len(active_y) > 0:
                # Average height location
                centroid_y = np.mean(active_y)
                # Invert height because image origin is top-left
                y_val = h - centroid_y
                y_values.append(y_val)
                x_indices.append(x)

        if len(x_indices) < 2:
            logger.warning("Lead mask contains insufficient traced pixels.")
            return np.zeros(w, dtype=np.float32)

        # Interpolate coordinates for columns containing gaps/no pixels
        full_x = np.arange(w)
        interp_fn = interp1d(x_indices, y_values, kind="linear", fill_value="extrapolate")
        reconstructed = interp_fn(full_x)

        # Apply smoothing
        if self.smoothing_window > 1:
            reconstructed = pd.Series(reconstructed).rolling(
                window=self.smoothing_window, min_periods=1, center=True
            ).mean().to_numpy()

        # Calibration: baseline normalization and physical unit conversion (mV)
        # (Using simulated scale: 50 pixels represents approx. 1 mV)
        calibrated = (reconstructed - np.median(reconstructed)) / 50.0

        return calibrated.astype(np.float32)
