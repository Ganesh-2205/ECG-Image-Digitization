from pathlib import Path
import pandas as pd

# ====================================
# Change this path if your dataset moves
# ====================================

DATASET_ROOT = Path("/content/drive/MyDrive/content/PTBXL_Images")

records = []

for image_path in DATASET_ROOT.rglob("*.png"):

    stem = image_path.stem.replace("-0", "")

    hea = image_path.with_name(stem + ".hea")
    dat = image_path.with_name(stem + ".dat")

    if hea.exists() and dat.exists():

        records.append({
            "image": str(image_path),
            "hea": str(hea),
            "dat": str(dat)
        })

df = pd.DataFrame(records)

print(df.head())

print(f"\nTotal Samples : {len(df)}")

df.to_csv("dataset_index.csv", index=False)
 
print("\nSaved dataset_index.csv")