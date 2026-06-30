"""
Orchestrates deep learning model training pipelines.
"""

from typing import List, Optional
import tensorflow as tf
from ..models.unet import BaseModel
from ..utils.config import Config
from ..utils.logger import SetupLogger
from .callbacks import CallbackBuilder

logger = SetupLogger.get_logger(__name__)


class ECGTrainer:
    """
    Coordinates training, compile configurations, learning-rate schedules,
    fitting loops, and validation tracking.
    """

    def __init__(self, config: Config, model_wrapper: BaseModel) -> None:
        """
        Initializes the trainer.

        Args:
            config: Global configuration object.
            model_wrapper: Instance of BaseModel (e.g. UNet or future Transformer).
        """
        self.config = config
        self.model_wrapper = model_wrapper
        self.epochs: int = self.config.get("training.epochs", 50)
        self.learning_rate: float = self.config.get("training.learning_rate", 0.001)

    def compile_model(
        self,
        loss: tf.keras.losses.Loss,
        metrics: List[tf.keras.metrics.Metric]
    ) -> None:
        """
        Compiles the model with specified loss functions and performance metrics.

        Args:
            loss: Subclass of tf.keras.losses.Loss.
            metrics: List of tf.keras.metrics.Metric indicators.
        """
        if self.model_wrapper.model is None:
            self.model_wrapper.build_model()
            
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model_wrapper.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        logger.info("Model compiled successfully for training.")

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.callbacks.History:
        """
        Executes the optimization loop on the compiled model.

        Args:
            train_dataset: Training tf.data.Dataset.
            val_dataset: Validation tf.data.Dataset.
            callbacks: List of Keras training callbacks.

        Returns:
            The training history object.
        """
        if self.model_wrapper.model is None or not self.model_wrapper.model.built:
            raise RuntimeError("Model must be compiled before fitting.")

        if callbacks is None:
            # Build standard callbacks automatically from configuration
            cb_builder = CallbackBuilder(self.config)
            callbacks = cb_builder.build_default_callbacks()

        logger.info(f"Starting training on model for {self.epochs} epochs...")
        history = self.model_wrapper.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=self.config.raw.get("verbose", 1)
        )
        logger.info("Model training loop completed.")
        return history
