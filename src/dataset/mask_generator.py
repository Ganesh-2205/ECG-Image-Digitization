import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class ECGMaskGenerator:
    """
    Generate binary segmentation masks from ECG images.
    """

    def __init__(self, input_dir, output_dir):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_mask(self, image_path):

        image = cv2.imread(str(image_path))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(
            gray,
            180,
            255,
            cv2.THRESH_BINARY_INV
        )

        kernel = np.ones((2, 2), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        return mask

    def generate(self):

        images = list(self.input_dir.rglob("*.png"))

        print(f"Found {len(images)} images")

        for image_path in tqdm(images):

            mask = self.create_mask(image_path)

            output_path = self.output_dir / image_path.name

            cv2.imwrite(
                str(output_path),
                mask
            )

        print("Mask generation completed.")  