import os
from datasets import load_dataset
from tqdm import tqdm
import sys
import time

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
TOTAL_SAMPLES = sum(SAMPLES_PER_CLASS.values())
SEED = 42 # for reproducibility

# Local target directory
DATA_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
RAW_DIRECTORY = os.path.join(DATA_DIRECTORY, 'raw')
os.makedirs(RAW_DIRECTORY, exist_ok=True)
# --- End Configuration ---

def download_and_save_class(class_label, num_samples, base_dir):
    """
    Downloads, filters, and saves samples for a single class to avoid large downloads.
    """
    print(f"\n--- Processing class: '{class_label}' ---")
    print(f"Attempting to download {num_samples} samples.")

    try:
        # Load the full dataset metadata in streaming mode to avoid downloading everything
        dataset = load_dataset(DATASET_NAME, split='train', streaming=True)

        # Filter for the specific class by converting integer label to string
        class_subset = dataset.filter(lambda x: dataset.features['label'].int2str(x['label']) == class_label)

        # Take the required number of samples
        class_subset_sample = class_subset.take(num_samples)

        # Create a directory for the class if it doesn't exist
        class_dir = os.path.join(base_dir, class_label)
        os.makedirs(class_dir, exist_ok=True)

        saved_count = 0
        for sample in tqdm(class_subset_sample, desc=f"Saving '{class_label}' images"):
            rgb_image = sample['rgb']
            
            # The dataset doesn't provide original filenames, so we create them.
            # We can use a hash of the image data to create a unique name.
            img_hash = hash(rgb_image.tobytes())
            img_filename = f"{img_hash}.png"
            img_path = os.path.join(class_dir, img_filename)
            
            if not os.path.exists(img_path):
                rgb_image.save(img_path)
            
            saved_count += 1
        
        print(f"Successfully saved {saved_count} samples for class '{class_label}'.")
        return saved_count

    except Exception as e:
        print(f"  [ERROR] Failed to process class '{class_label}': {e}")
        return 0

if __name__ == '__main__':
    print(f"--- 3D-ADAM Stratified Subset Creation ---")
    total_saved = 0
    
    # Process each class individually to manage API requests
    for class_label, num_samples in SAMPLES_PER_CLASS.items():
        count = download_and_save_class(class_label, num_samples, RAW_DIRECTORY)
        total_saved += count
        # Optional: Add a small delay between processing classes if rate limiting persists
        time.sleep(5)

    print("\n--- Data processing complete! ---")
    print(f"Total images saved: {total_saved}")
    print(f"Data is located in: {RAW_DIRECTORY}")