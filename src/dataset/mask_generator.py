import cv2
import numpy as np


class ECGMaskGenerator:
    """
    Generates binary masks from ECG images.
    """

    def __init__(self, threshold=180):
        self.threshold = threshold

    def generate_mask(self, image):

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binary inverse threshold
        _, mask = cv2.threshold(
            gray,
            self.threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Remove tiny noisy regions
        kernel = np.ones((2, 2), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        return mask

    def save_mask(self, mask, save_path):
        cv2.imwrite(save_path, mask)