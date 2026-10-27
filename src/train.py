import os
import time
import pickle
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    from src.logger import logger
except ImportError:
    from logger import logger


class ModelTrainer:
    """
    Trains TF-IDF vectorizer and Nearest Neighbors model on processed product text,
    saves artifacts, and logs experiment parameters with MLflow.
    """

    def __init__(
        self,
        model_dir: str = "artifacts/models",
        data_dir: str = "artifacts/data",
        max_features: int = 10000,
        n_neighbors: int = 11,
        metric: str = "cosine",
    ):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.max_features = max_features
        self.n_neighbors = n_neighbors
        self.metric = metric

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def train(self, df: pd.DataFrame, enable_mlflow: bool = True) -> Dict[str, Any]:
        """
        Trains TF-IDF representation and Nearest Neighbors indexing on df['combined_text'].
        Saves models and processed dataset in both Parquet and CSV formats.
        """
        start_time = time.time()
        logger.info(f"Initiating training on {len(df)} products...")

        # 1. Save processed datasets for serving
        parquet_path = os.path.join(self.data_dir, "processed_data.parquet")
        csv_path = os.path.join(self.data_dir, "processed_data.csv")

        logger.info(f"Saving processed data to {parquet_path} and {csv_path}...")
        df.to_parquet(parquet_path, compression="zstd", index=False)
        # Also write lightweight CSV if needed
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)

        # 2. Fit TF-IDF Vectorizer
        logger.info(f"Fitting TF-IDF Vectorizer (max_features={self.max_features})...")
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
        )
        tfidf_matrix = vectorizer.fit_transform(df["combined_text"].fillna("").astype(str))
        logger.info(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

        # 3. Fit Nearest Neighbors
        logger.info(
            f"Fitting Nearest Neighbors model (n_neighbors={self.n_neighbors}, metric={self.metric})..."
        )
        nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm="brute",  # Optimal for sparse cosine similarity
        )
        nn_model.fit(tfidf_matrix)

        # 4. Serialize Model Artifacts
        nn_model_path = os.path.join(self.model_dir, "nn_model.pkl")
        vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
        tfidf_matrix_path = os.path.join(self.model_dir, "tfidf_matrix.pkl")

        with open(nn_model_path, "wb") as f:
            pickle.dump(nn_model, f)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        with open(tfidf_matrix_path, "wb") as f:
            pickle.dump(tfidf_matrix, f)

        training_duration = round(time.time() - start_time, 2)
        logger.info(f"Training completed successfully in {training_duration}s.")

        # 5. Optional MLflow Logging
        if enable_mlflow:
            self._log_to_mlflow(
                dataset_size=len(df),
                matrix_shape=tfidf_matrix.shape,
                duration=training_duration,
                nn_model_path=nn_model_path,
                vectorizer_path=vectorizer_path,
            )

        return {
            "nn_model": nn_model,
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "training_duration": training_duration,
            "dataset_size": len(df),
        }

    def _log_to_mlflow(
        self,
        dataset_size: int,
        matrix_shape: tuple,
        duration: float,
        nn_model_path: str,
        vectorizer_path: str,
    ):
        """Logs parameters and metrics to MLflow."""
        try:
            import mlflow

            mlflow.set_experiment("Product_Recommender")
            with mlflow.start_run(run_name="TFIDF_NearestNeighbors"):
                mlflow.log_param("max_features", self.max_features)
                mlflow.log_param("n_neighbors", self.n_neighbors)
                mlflow.log_param("metric", self.metric)
                mlflow.log_param("algorithm", "brute")

                mlflow.log_metric("dataset_size", dataset_size)
                mlflow.log_metric("matrix_rows", matrix_shape[0])
                mlflow.log_metric("matrix_cols", matrix_shape[1])
                mlflow.log_metric("training_duration_seconds", duration)

                if os.path.exists(nn_model_path):
                    mlflow.log_artifact(nn_model_path, artifact_path="models")
                if os.path.exists(vectorizer_path):
                    mlflow.log_artifact(vectorizer_path, artifact_path="models")
            logger.info("Logged parameters and metrics to MLflow successfully.")
        except Exception as e:
            logger.warning(f"MLflow tracking skipped: {e}")