"""
Dataset splitting utility.
"""

import os
from typing import List, Tuple
import numpy as np
from ..utils.config import Config
from ..utils.logger import SetupLogger

logger = SetupLogger.get_logger(__name__)


class DatasetSplitter:
    """
    Splits image and mask files into deterministic train, validation, and test sets.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the dataset splitter.

        Args:
            config: Global configuration object.
        """
        self.config = config
        self.seed: int = self.config.get("seed", 42)

    def split_data(
        self,
        images_dir: str,
        masks_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Tuple[
        Tuple[List[str], List[str]],
        Tuple[List[str], List[str]],
        Tuple[List[str], List[str]]
    ]:
        """
        Splits image-mask file pairs into sets.

        Args:
            images_dir: Directory containing ECG images.
            masks_dir: Directory containing ECG masks.
            train_ratio: Proportion of files for training.
            val_ratio: Proportion of files for validation.

        Returns:
            Nested tuple structure: ((train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks))
        """
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            logger.warning("Directory paths do not exist. Returning empty splits.")
            return ([], []), ([], []), ([], [])

        all_imgs = sorted(
            [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".png")]
        )
        all_masks = sorted(
            [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(".png")]
        )

        if len(all_imgs) != len(all_masks):
            raise ValueError(
                f"Mismatch in image and mask counts: {len(all_imgs)} images vs {len(all_masks)} masks."
            )

        num_samples = len(all_imgs)
        rng = np.random.default_rng(self.seed)
        indices = np.arange(num_samples)
        rng.shuffle(indices)

        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_imgs = [all_imgs[i] for i in train_idx]
        train_masks = [all_masks[i] for i in train_idx]

        val_imgs = [all_imgs[i] for i in val_idx]
        val_masks = [all_masks[i] for i in val_idx]

        test_imgs = [all_imgs[i] for i in test_idx]
        test_masks = [all_masks[i] for i in test_idx]

        logger.info(
            f"Dataset split completed. Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}"
        )

        return (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks)
