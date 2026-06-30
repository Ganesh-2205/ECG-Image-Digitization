"""
Keras callback builder utilities.
"""

import os
from typing import List
import tensorflow as tf
from ..utils.config import Config
from ..utils.logger import SetupLogger

logger = SetupLogger.get_logger(__name__)


class CallbackBuilder:
    """
    Constructs and configures Keras callbacks for training monitoring and logging.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the CallbackBuilder.

        Args:
            config: Global configuration object.
        """
        self.config = config
        self.saved_models_dir: str = self.config.get("paths.saved_models_dir", "saved_models")
        self.outputs_dir: str = self.config.get("paths.outputs_dir", "outputs")

    def build_default_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Constructs standard training callbacks:
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and TensorBoard.

        Returns:
            A list of initialized Keras Callback instances.
        """
        callbacks: List[tf.keras.callbacks.Callback] = []

        # Model Checkpoint Callback
        checkpoint_path = os.path.join(self.saved_models_dir, "best_model.keras")
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint_cb)

        # Early Stopping Callback
        early_stopping_patience = self.config.get("training.early_stopping_patience", 10)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_cb)

        # Learning Rate Reduction Callback
        reduce_lr_patience = self.config.get("training.reduce_lr_patience", 5)
        reduce_lr_factor = self.config.get("training.reduce_lr_factor", 0.5)
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr_cb)

        # TensorBoard Logging Callback
        log_dir = os.path.join(self.outputs_dir, "logs")
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        callbacks.append(tensorboard_cb)

        logger.info(f"Initialized {len(callbacks)} default training callbacks.")
        return callbacks
