"""
Preprocessing and data augmentation module for ECG paper images and masks.
"""

from typing import Tuple
import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
from ..utils.config import Config
from ..utils.logger import SetupLogger

logger = SetupLogger.get_logger(__name__)


class ECGImagePreprocessor:
    """
    Enforces standardized image sizing, pixel normalization, and data augmentations.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the preprocessor with configurations.

        Args:
            config: Global configuration object.
        """
        self.config = config
        self.target_width: int = self.config.get("preprocess.target_width", 512)
        self.target_height: int = self.config.get("preprocess.target_height", 512)
        
        # Setup albumentations pipeline
        self.aug_pipeline = A.Compose([
            A.Rotate(limit=self.config.get("preprocess.augmentation.rotation_range", 5), p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.2),
        ])

    def preprocess_image_tf(self, img_bytes: tf.Tensor) -> tf.Tensor:
        """
        Decodes, resizes, and normalizes an image using TF graph operations.

        Args:
            img_bytes: Input raw image bytes.

        Returns:
            Preprocessed float32 image tensor.
        """
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # Normalizes to [0, 1]
        img = tf.image.resize(img, [self.target_height, self.target_width])
        return img

    def preprocess_mask_tf(self, mask_bytes: tf.Tensor) -> tf.Tensor:
        """
        Decodes, resizes, and binarizes a mask using TF graph operations.

        Args:
            mask_bytes: Input raw mask bytes.

        Returns:
            Binarized float32 mask tensor.
        """
        mask = tf.image.decode_png(mask_bytes, channels=1)
        mask = tf.image.convert_image_dtype(mask, tf.float32)
        mask = tf.image.resize(
            mask, [self.target_height, self.target_width], method="nearest"
        )
        # Force strict binarization
        mask = tf.where(mask > 0.5, 1.0, 0.0)
        return mask

    def augment_np(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Albumentations augmentation pipeline to numpy arrays.

        Args:
            img: Input image numpy array.
            mask: Input mask numpy array.

        Returns:
            Augmented image and mask.
        """
        augmented = self.aug_pipeline(image=img, mask=mask)
        return augmented["image"], augmented["mask"]

    def augment_tf(self, img: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Wraps Albumentations numpy augmentation as a TensorFlow operation.

        Args:
            img: Input image tensor.
            mask: Input mask tensor.

        Returns:
            Augmented image and mask tensors.
        """
        def _aug_wrapper(image_np: np.ndarray, mask_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return self.augment_np(image_np, mask_np)

        aug_img, aug_mask = tf.py_function(
            func=_aug_wrapper,
            inp=[img, mask],
            Tout=[tf.float32, tf.float32]
        )
        
        # Restore shapes explicitly
        aug_img.set_shape([self.target_height, self.target_width, 3])
        aug_mask.set_shape([self.target_height, self.target_width, 1])
        
        return aug_img, aug_mask
