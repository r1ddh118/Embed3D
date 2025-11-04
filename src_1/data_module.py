# data_module.py
import os
import pickle
from datasets import load_dataset
import random

class AdamDataModule:
    def __init__(self, dataset_name="pmchard/3D-ADAM", subset_percent=10, cache_file="adam_5pct_subset.pkl", seed=42):
        self.dataset_name = dataset_name
        self.subset_percent = subset_percent
        self.cache_file = cache_file
        self.seed = seed

    def load_subset(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded cached subset from {self.cache_file}")
            return data

        print(f"Streaming and collecting {self.subset_percent}% of {self.dataset_name} ...")
        ds = load_dataset(self.dataset_name, split="train", streaming=True)
        # default total estimate; adjust if known
        total_est = 14120
        take_n = max(1, int(total_est * self.subset_percent / 100))
        data_list = list(ds.take(take_n))
        random.Random(self.seed).shuffle(data_list)
        with open(self.cache_file, "wb") as f:
            pickle.dump(data_list, f)
        print(f"Saved subset to {self.cache_file} ({len(data_list)} items)")
        return data_list
