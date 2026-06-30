"""
ECG Data Loader module using TensorFlow data pipeline.
"""

from typing import List, Tuple
import tensorflow as tf
from ..utils.config import Config
from ..utils.logger import SetupLogger
from .preprocessing import ECGImagePreprocessor

logger = SetupLogger.get_logger(__name__)


class ECGDataLoader:
    """
    Handles TensorFlow data pipelines for loading and feeding ECG images and masks.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the data loader.

        Args:
            config: Global configuration object.
        """
        self.config = config
        self.batch_size: int = self.config.get("preprocess.batch_size", 16)
        self.preprocessor = ECGImagePreprocessor(self.config)

    def _parse_function(
        self, img_path: tf.Tensor, mask_path: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        TensorFlow mapping function to load and preprocess a single image/mask pair.

        Args:
            img_path: Tensor containing the image file path.
            mask_path: Tensor containing the mask file path.

        Returns:
            Preprocessed image and mask tensors.
        """
        # Load raw files
        img_bytes = tf.io.read_file(img_path)
        mask_bytes = tf.io.read_file(mask_path)

        # Preprocess using Tensorflow ops inside the graph or wrapping numpy
        img = self.preprocessor.preprocess_image_tf(img_bytes)
        mask = self.preprocessor.preprocess_mask_tf(mask_bytes)

        return img, mask

    def get_dataset(
        self,
        img_paths: List[str],
        mask_paths: List[str],
        shuffle: bool = True,
        augment: bool = False,
    ) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from lists of image and mask file paths.

        Args:
            img_paths: List of file paths to ECG images.
            mask_paths: List of file paths to ECG masks.
            shuffle: Whether to shuffle the dataset.
            augment: Whether to apply online data augmentation.

        Returns:
            An optimized tf.data.Dataset instance.
        """
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
        
        # Load and parse images/masks
        dataset = dataset.map(
            self._parse_function, num_parallel_calls=tf.data.AUTOTUNE
        )

        if augment:
            # Applies Albumentations augmentations (via py_function) or TF augmentations
            dataset = dataset.map(
                lambda x, y: (self.preprocessor.augment_tf(x, y)),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if shuffle:
            buffer_size = self.config.get("preprocess.shuffle_buffer", 1000)
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
