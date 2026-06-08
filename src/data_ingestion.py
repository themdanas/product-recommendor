import os
import pandas as pd

class DataIngestion:

    def __init__(self):
        self.raw_data_path = "artifacts/data/raw_data.csv"

    def ingest_data(self):

        if os.path.exists(self.raw_data_path):

            print("Loading local dataset...")

            return pd.read_csv(self.raw_data_path,
                               low_memory=False).drop( columns=["subtitle","author", "bought_together"],
                               errors="ignore"
                               )

        # download dataset only if file doesn't exist