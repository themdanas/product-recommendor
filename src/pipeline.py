from typing import Dict, Any, Optional
from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.feature_engineering import FeatureEngineering
from src.train import ModelTrainer
from src.evaluate import Evaluator
from src.logger import logger


class MLPipeline:
    """
    Coordinates the complete End-to-End ML Pipeline:
    Ingestion -> Validation -> Feature Engineering -> Training -> Evaluation.
    """

    def __init__(
        self,
        raw_data_path: str = "artifacts/data/raw_data.csv",
        processed_data_path: str = "artifacts/data/processed_data.parquet",
        model_dir: str = "artifacts/models",
    ):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.model_dir = model_dir

        self.ingestion = DataIngestion(
            raw_data_path=self.raw_data_path,
            processed_data_path=self.processed_data_path,
        )
        self.validator = DataValidation()
        self.fe = FeatureEngineering()
        self.trainer = ModelTrainer(model_dir=self.model_dir)
        self.evaluator = Evaluator()

    def run_pipeline(
        self,
        min_ratings: int = 5,
        max_features: int = 10000,
        enable_mlflow: bool = True,
    ) -> Dict[str, Any]:
        """Runs all pipeline stages sequentially."""
        logger.info("=== Starting E-Commerce Recommender ML Pipeline ===")

        # Stage 1: Data Ingestion
        logger.info("--> Stage 1: Data Ingestion")
        df = self.ingestion.ingest_data()
        logger.info(f"Ingested dataset with {len(df)} rows.")

        # Stage 2: Data Validation
        logger.info("--> Stage 2: Data Validation")
        validation_report = self.validator.validate(df)

        # Stage 3: Feature Engineering
        logger.info("--> Stage 3: Feature Engineering")
        df_processed = self.fe.create_combined_text(df, min_ratings=min_ratings)

        # Stage 4: Model Training
        logger.info("--> Stage 4: Model Training")
        train_results = self.trainer.train(df_processed, enable_mlflow=enable_mlflow)

        # Stage 5: Sample Evaluation
        logger.info("--> Stage 5: Model Evaluation")
        sample_title = df_processed.iloc[0]["title"]
        logger.info(f"Pipeline successfully completed for {len(df_processed)} products.")

        return {
            "validation_report": validation_report,
            "training_results": train_results,
            "product_count": len(df_processed),
            "sample_product": sample_title,
        }
