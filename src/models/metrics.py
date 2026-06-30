"""
Custom evaluation metrics for model verification.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


class DiceCoefficient(tf.keras.metrics.Metric):
    """
    Keras-compatible stateful metric for computing Dice overlap.
    """

    def __init__(self, smooth: float = 1e-5, name: str = "dice_coefficient", **kwargs) -> None:
        """
        Initializes the Dice Coefficient metric.

        Args:
            smooth: Smoothing factor.
            name: Metric identifier.
        """
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.total_intersection = self.add_weight(name="intersection", initializer="zeros")
        self.total_sum = self.add_weight(name="total_sum", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None) -> None:
        """
        Updates the running metric calculations.
        """
        y_true_f = tf.cast(K.flatten(y_true), tf.float32)
        y_pred_f = tf.cast(K.flatten(y_pred), tf.float32)
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union_sum = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        self.total_intersection.assign_add(intersection)
        self.total_sum.assign_add(union_sum)

    def result(self) -> tf.Tensor:
        """
        Returns the final calculated Dice coefficient.
        """
        return (2.0 * self.total_intersection + self.smooth) / (self.total_sum + self.smooth)

    def reset_state(self) -> None:
        """
        Resets running weights between validation or epoch thresholds.
        """
        self.total_intersection.assign(0.0)
        self.total_sum.assign(0.0)

    def reset_states(self) -> None:
        self.reset_state()
