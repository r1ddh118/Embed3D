import os
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import io

# --- Configuration ---
DATASET_NAME = "pmchard/3D-ADAM"
SAMPLES_PER_CLASS = {
    "good": 300,
    "crease": 60,
    "dent": 60,
    "gap": 60,
    "hole": 60,
    "scratch": 60,
}

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
RAW_DIR = os.path.join(BASE_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Initialize counters for each class
saved_count = {cls: 0 for cls in SAMPLES_PER_CLASS}


def get_class_from_path(path: str):
    """
    Extracts the class name from an image path.
    """
    parts = path.replace("\\", "/").split("/")
    for c in SAMPLES_PER_CLASS.keys():
        if c in parts:
            return c
    return None


def save_image(rgb_data, save_path):
    """Save RGB image from Hugging Face sample."""
    try:
        if isinstance(rgb_data, Image.Image):
            img = rgb_data
        elif isinstance(rgb_data, (bytes, bytearray)):
            img = Image.open(io.BytesIO(rgb_data))
        elif hasattr(rgb_data, "numpy"):
            from PIL import Image
            img = Image.fromarray(rgb_data.numpy())
        else:
            return False
        img.save(save_path)
        return True
    except Exception as e:
        print(f"[WARN] Could not save image: {e}")
        return False


if __name__ == "__main__":
    print(f"--- Loading dataset: {DATASET_NAME} ---")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    print("✅ Dataset loaded.")

    # Pre-create folders for each class
    for cls in SAMPLES_PER_CLASS:
        os.makedirs(os.path.join(RAW_DIR, cls), exist_ok=True)

    print("--- Streaming dataset and saving samples ---")
    for sample in tqdm(dataset, desc="Streaming dataset"):
        # Stop early if all class quotas reached
        if all(saved_count[cls] >= SAMPLES_PER_CLASS[cls] for cls in SAMPLES_PER_CLASS):
            break

        # Try to get path
        path_fields = [k for k in sample.keys() if "path" in k or "file" in k]
        file_path = None
        if path_fields:
            file_path = sample[path_fields[0]]
        elif "rgb" in sample and hasattr(sample["rgb"], "filename"):
            file_path = sample["rgb"].filename
        else:
            file_path = str(sample)

        # Infer class
        cls = get_class_from_path(str(file_path))
        if not cls:
            continue

        # Check quota
        if saved_count[cls] >= SAMPLES_PER_CLASS[cls]:
            continue

        # Save image
        save_path = os.path.join(RAW_DIR, cls, f"{saved_count[cls]}.png")
        if save_image(sample["rgb"], save_path):
            saved_count[cls] += 1

    print("\nFinished saving samples!")
    for cls in SAMPLES_PER_CLASS:
        print(f"{cls}: {saved_count[cls]} images saved")
    print(f"Saved in: {RAW_DIR}")
