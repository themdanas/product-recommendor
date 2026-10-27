import pandas as pd
from typing import Dict, Any
from src.logger import logger


class DataValidation:
    """
    Validates the structure and quality constraints of the dataset
    before feature engineering and model training.
    """

    REQUIRED_COLUMNS = [
        "title",
        "description",
        "features",
        "categories",
        "parent_asin",
        "average_rating",
        "rating_number",
    ]

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs schema and data quality validations on the DataFrame.
        Returns a summary report dictionary if valid, or raises ValueError/AssertionError.
        """
        logger.info("Starting data validation checks...")

        # 1. Required Columns Check
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Data validation failed: Missing required columns: {missing_cols}")

        # 2. Ratings Range Check (0 to 5)
        # Note: drop NaN or non-numeric if any
        valid_ratings = df["average_rating"].dropna()
        if not valid_ratings.between(0, 5).all():
            min_r = valid_ratings.min()
            max_r = valid_ratings.max()
            raise ValueError(f"Ratings outside valid range [0, 5]: Min={min_r}, Max={max_r}")

        # 3. Rating Count (Non-negative)
        valid_counts = df["rating_number"].dropna()
        if not (valid_counts >= 0).all():
            raise ValueError("Found negative rating counts in 'rating_number'")

        # 4. Check Parent ASIN presence
        null_asins = df["parent_asin"].isnull().sum()
        if null_asins > 0:
            logger.warning(f"Found {null_asins} records with missing parent_asin.")

        # 5. Check Duplicates
        duplicate_asins = df["parent_asin"].duplicated().sum()
        if duplicate_asins > 0:
            logger.warning(f"Found {duplicate_asins} duplicate parent_asin entries.")

        summary = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "duplicate_asins": int(duplicate_asins),
            "rating_range": [float(valid_ratings.min()), float(valid_ratings.max())],
            "average_rating_mean": round(float(valid_ratings.mean()), 2),
            "passed": True,
        }
        logger.info(f"Data validation successful: {summary['total_rows']} valid rows.")
        return summary