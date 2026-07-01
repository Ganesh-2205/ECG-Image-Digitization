from pathlib import Path
import cv2

from src.dataset.mask_generator import ECGMaskGenerator

# -----------------------------
# Change this path if needed
# -----------------------------
IMAGE_ROOT = Path("dataset/images")
MASK_ROOT = Path("dataset/masks")

MASK_ROOT.mkdir(parents=True, exist_ok=True)

generator = ECGMaskGenerator()

images = list(IMAGE_ROOT.rglob("*.png"))

print(f"Found {len(images)} images")

for img_path in images:

    image = cv2.imread(str(img_path))

    mask = generator.generate_mask(image)

    save_name = img_path.stem + "_mask.png"

    generator.save_mask(
        mask,
        MASK_ROOT / save_name
    )

print("All masks generated successfully.")