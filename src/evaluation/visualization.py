"""
Visualizations: graph plotting, overlay traces, and training history outputs.
"""

import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from ..utils.config import Config
from ..utils.logger import SetupLogger

logger = SetupLogger.get_logger(__name__)


class SignalVisualizer:
    """
    Renders diagnostic charts including overlays, segmentation masks, and training metrics.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the visualizer.

        Args:
            config: Global configuration object.
        """
        self.config = config
        self.plots_dir: str = self.config.get("paths.plots_dir", "outputs/plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_overlay(
        self, gt: np.ndarray, recon: np.ndarray, lead_name: str, file_name: str
    ) -> None:
        """
        Plots the reconstructed signal overlaid on the ground-truth digital signal.

        Args:
            gt: Ground-truth 1D signal.
            recon: Reconstructed 1D signal.
            lead_name: Name of the ECG lead.
            file_name: Output filename.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(gt, label="Ground Truth", color="blue", alpha=0.7)
        plt.plot(recon, label="Reconstructed", color="red", linestyle="--", alpha=0.8)
        plt.title(f"ECG Trace Reconstruction - Lead {lead_name}")
        plt.xlabel("Samples")
        plt.ylabel("Voltage (mV)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(self.plots_dir, file_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved overlay graph to {save_path}")

    def plot_history(self, history: Dict[str, List[float]], file_name: str) -> None:
        """
        Plots training and validation loss curves.

        Args:
            history: Dictionary of metrics from Keras History.history.
            file_name: Output filename.
        """
        plt.figure(figsize=(8, 5))
        for key, values in history.items():
            plt.plot(values, label=key)

        plt.title("Model Training Performance History")
        plt.xlabel("Epochs")
        plt.ylabel("Metric Values")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(self.plots_dir, file_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved training history plot to {save_path}")
