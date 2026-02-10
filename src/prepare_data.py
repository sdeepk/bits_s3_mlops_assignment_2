import os
import random
import zipfile

from PIL import Image
from tqdm import tqdm

RAW_ZIP = "data/raw/cats_dogs.zip"
EXTRACT_DIR = "data/raw"
SOURCE_DIR = "data/raw/PetImages"
OUTPUT_DIR = "data/processed"

IMG_SIZE = 224
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}

random.seed(42)

CLASS_MAP = {
    "Cat": "cats",
    "Dog": "dogs"
}

def extract_zip():
    with zipfile.ZipFile(RAW_ZIP, "r") as z:
        z.extractall(EXTRACT_DIR)

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False

def prepare():
    extract_zip()

    for split in SPLIT:
        for cls in CLASS_MAP.values():
            os.makedirs(f"{OUTPUT_DIR}/{split}/{cls}", exist_ok=True)

    for src_cls, dst_cls in CLASS_MAP.items():
        src_path = os.path.join(SOURCE_DIR, src_cls)
        images = os.listdir(src_path)
        random.shuffle(images)

        valid_images = []
        for img_name in images:
            img_path = os.path.join(src_path, img_name)
            if is_valid_image(img_path):
                valid_images.append(img_name)

        n = len(valid_images)
        train_end = int(SPLIT["train"] * n)
        val_end = train_end + int(SPLIT["val"] * n)

        splits = {
            "train": valid_images[:train_end],
            "val": valid_images[train_end:val_end],
            "test": valid_images[val_end:]
        }

        for split, files in splits.items():
            for f in tqdm(files, desc=f"{dst_cls}-{split}"):
                src_file = os.path.join(src_path, f)
                img = Image.open(src_file).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img.save(f"{OUTPUT_DIR}/{split}/{dst_cls}/{f}")

    print("Preprocessing complete: data/processed ready!!")

if __name__ == "__main__":
    prepare()
