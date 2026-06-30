"""
Quantitative evaluation of reconstructed ECG signals against ground truth.
"""

from typing import Dict
import numpy as np
from scipy.stats import pearsonr
from ..utils.config import Config
from ..utils.logger import SetupLogger

logger = SetupLogger.get_logger(__name__)


class ECGSignalEvaluator:
    """
    Computes statistical similarity indices comparing original signals
    and post-processed reconstruction results.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the evaluator.

        Args:
            config: Global configuration object.
        """
        self.config = config
        self.metrics_to_compute = self.config.get("evaluation.metrics", ["rmse", "pearson_corr", "snr"])

    def rmse(self, gt: np.ndarray, recon: np.ndarray) -> float:
        """
        Calculates Root Mean Squared Error.
        """
        return float(np.sqrt(np.mean((gt - recon) ** 2)))

    def pearson_correlation(self, gt: np.ndarray, recon: np.ndarray) -> float:
        """
        Calculates Pearson Correlation Coefficient.
        """
        if np.std(gt) == 0 or np.std(recon) == 0:
            return 0.0
        corr, _ = pearsonr(gt, recon)
        return float(corr)

    def snr(self, gt: np.ndarray, recon: np.ndarray) -> float:
        """
        Calculates Signal-to-Noise Ratio (SNR) in dB.
        """
        signal_power = np.mean(gt ** 2)
        noise_power = np.mean((gt - recon) ** 2)
        if noise_power == 0:
            return float("inf")
        return float(10 * np.log10(signal_power / noise_power))

    def evaluate_lead(self, gt: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
        """
        Computes all requested metrics for a single lead trace.

        Args:
            gt: Ground-truth 1D signal.
            recon: Reconstructed 1D signal.

        Returns:
            Dictionary mapping metric names to value floats.
        """
        # Ensure identical sizes by cropping/padding
        min_len = min(len(gt), len(recon))
        gt_aligned = gt[:min_len]
        recon_aligned = recon[:min_len]

        # Align baselines (zero-mean normalization)
        gt_aligned = gt_aligned - np.mean(gt_aligned)
        recon_aligned = recon_aligned - np.mean(recon_aligned)
        
        # Scale alignment if standard deviation exists
        if np.std(recon_aligned) > 0:
            recon_aligned = recon_aligned * (np.std(gt_aligned) / np.std(recon_aligned))

        metrics: Dict[str, float] = {}
        if "rmse" in self.metrics_to_compute:
            metrics["rmse"] = self.rmse(gt_aligned, recon_aligned)
        if "pearson_corr" in self.metrics_to_compute:
            metrics["pearson_corr"] = self.pearson_correlation(gt_aligned, recon_aligned)
        if "snr" in self.metrics_to_compute:
            metrics["snr"] = self.snr(gt_aligned, recon_aligned)

        return metrics
