import re
import numpy as np
import pandas as pd
from typing import Any
from src.logger import logger


class FeatureEngineering:
    """
    Cleans textual metadata and constructs a unified representation for TF-IDF vectorization.
    """

    def clean_text(self, text: Any) -> str:
        """Sanitizes text by lowercasing and removing special punctuation/whitespace."""
        if text is None or pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove special characters
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _stringify_field(self, val: Any) -> str:
        """Converts lists, arrays or scalar objects to clean whitespace-separated strings."""
        if val is None:
            return ""
        if isinstance(val, (list, tuple, np.ndarray)):
            return " ".join([str(item) for item in val if item is not None])
        if pd.isna(val):
            return ""
        s = str(val).strip()
        if s.lower() in ("nan", "none", "[]", "{}"):
            return ""
        return s

    def create_combined_text(self, df: pd.DataFrame, min_ratings: int = 5) -> pd.DataFrame:
        """
        Filters products with at least `min_ratings` reviews and concatenates
        title, features, categories, and description into a single 'combined_text' feature.
        """
        logger.info(f"Filtering dataset by min_ratings >= {min_ratings}...")
        initial_len = len(df)
        df_filtered = df[df["rating_number"].fillna(0) >= min_ratings].copy()
        logger.info(f"Retained {len(df_filtered)} / {initial_len} products.")

        logger.info("Combining text features: title, features, categories, description...")
        title = df_filtered["title"].apply(self._stringify_field)
        features = df_filtered["features"].apply(self._stringify_field) if "features" in df_filtered else ""
        categories = df_filtered["categories"].apply(self._stringify_field) if "categories" in df_filtered else ""
        description = df_filtered["description"].apply(self._stringify_field) if "description" in df_filtered else ""

        df_filtered["combined_text"] = title + " " + features + " " + categories + " " + description
        df_filtered["combined_text"] = df_filtered["combined_text"].apply(self.clean_text)

        logger.info("Feature engineering completed successfully.")
        return df_filtered