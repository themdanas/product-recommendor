import os
import pandas as pd
from src.logger import logger


class DataIngestion:
    """
    Handles loading raw datasets from local disk or downloading
    from the Hugging Face Hub (McAuley-Lab/Amazon-Reviews-2023).
    """

    def __init__(
        self,
        raw_data_path: str = "artifacts/data/raw_data.csv",
        processed_data_path: str = "artifacts/data/processed_data.parquet",
    ):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.output_dir = os.path.dirname(self.raw_data_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def ingest_data(self) -> pd.DataFrame:
        """
        Loads the appliance metadata dataset.
        Checks local processed and raw files first, falling back to Hugging Face.
        """
        # 1. Check if processed parquet already exists
        if os.path.exists(self.processed_data_path):
            logger.info(f"Loading existing processed data from {self.processed_data_path}...")
            return pd.read_parquet(self.processed_data_path)

        # 2. Check if local raw CSV exists
        if os.path.exists(self.raw_data_path):
            logger.info(f"Loading local raw dataset from {self.raw_data_path}...")
            df = pd.read_csv(self.raw_data_path, low_memory=False)
            return self._clean_raw_columns(df)

        # 3. Check if processed CSV exists
        csv_processed = os.path.join(self.output_dir, "processed_data.csv")
        if os.path.exists(csv_processed):
            logger.info(f"Loading local processed CSV from {csv_processed}...")
            df = pd.read_csv(csv_processed, low_memory=False)
            return self._clean_raw_columns(df)

        # 4. Download from Hugging Face if not available locally
        logger.info("Local dataset not found. Downloading 'raw_meta_Appliances' from Hugging Face...")
        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                "raw_meta_Appliances",
                split="full",
            )
            df = dataset.to_pandas()
            logger.info(f"Downloaded {len(df)} rows from Hugging Face.")

            df = self._clean_raw_columns(df)

            # Cache locally as raw_data.csv
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Cached raw dataset to {self.raw_data_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to download dataset from Hugging Face: {e}")
            raise RuntimeError(
                f"Data ingestion failed: Could not locate local data and remote download failed: {e}"
            )

    def _clean_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops unused empty columns from the raw dataset."""
        drop_cols = ["subtitle", "author", "bought_together"]
        return df.drop(columns=drop_cols, errors="ignore")