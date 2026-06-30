"""
Custom loss functions for semantic segmentation.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


class DiceLoss(tf.keras.losses.Loss):
    """
    Dice loss class to maximize overlay overlap in semantic segmentations.
    """

    def __init__(self, smooth: float = 1e-5, name: str = "dice_loss") -> None:
        """
        Initializes DiceLoss.

        Args:
            smooth: Smoothing factor to avoid division by zero.
            name: Loss name.
        """
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the Dice loss value.
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)
        return 1.0 - dice


class DiceBCELoss(tf.keras.losses.Loss):
    """
    Hybrid loss combining Binary Crossentropy and Dice Loss.
    """

    def __init__(self, smooth: float = 1e-5, name: str = "dice_bce_loss") -> None:
        """
        Initializes DiceBCELoss.

        Args:
            smooth: Smoothing factor.
            name: Loss name.
        """
        super().__init__(name=name)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the hybrid loss value.
        """
        bce = self.bce_loss(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        return bce + dice
