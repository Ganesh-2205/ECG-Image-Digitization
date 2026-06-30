"""
Handles inference predictions on test ECG images using trained models.
"""

from typing import Union
import cv2
import numpy as np
import tensorflow as tf
from ..models.unet import BaseModel
from ..utils.config import Config
from ..utils.logger import SetupLogger

logger = SetupLogger.get_logger(__name__)


class ECGPredictor:
    """
    Inference executor applying trained semantic segmentation weights
    on individual paper ECG images to yield probability maps.
    """

    def __init__(self, config: Config, model_wrapper: BaseModel, model_path: str) -> None:
        """
        Initializes the predictor.

        Args:
            config: Global configuration object.
            model_wrapper: Instance of BaseModel.
            model_path: Path to the trained checkpoint file.
        """
        self.config = config
        self.model_wrapper = model_wrapper
        self.model_path = model_path
        
        self.target_width = self.config.get("preprocess.target_width", 512)
        self.target_height = self.config.get("preprocess.target_height", 512)
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads model architecture and restores weights.
        """
        if self.model_wrapper.model is None:
            self.model_wrapper.build_model()
        self.model_wrapper.load_checkpoint(self.model_path)
        logger.info("Predictor initialized with trained model.")

    def predict(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Applies inference on a single input image.

        Args:
            image: Either a file path string or a loaded NumPy array.

        Returns:
            Segmentation probability map as a 2D float32 numpy array.
        """
        if isinstance(image, str):
            # Load and convert to RGB
            img_np = cv2.imread(image)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img_np = image

        # Resize and scale to [0, 1]
        img_resized = cv2.resize(img_np, (self.target_width, self.target_height))
        img_tensor = img_resized.astype(np.float32) / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_tensor, axis=0)

        # Run model inference
        predictions = self.model_wrapper.model.predict(img_batch, verbose=0)
        
        # Remove batch and channel dimensions from result
        prob_map = np.squeeze(predictions, axis=(0, -1))
        
        return prob_map
