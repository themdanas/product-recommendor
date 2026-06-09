import os
import pickle
from logger import logger

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class ModelTrainer:

    def __init__(self):

        self.model_dir = "artifacts/models"

        os.makedirs(
            self.model_dir,
            exist_ok=True
        )

    def train(self, df):

        df.to_csv(
        "artifacts/data/processed_data.csv",
        index=False
        )

        print("Creating TF-IDF vectors...")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000
        )

        tfidf_matrix = vectorizer.fit_transform(
            df["combined_text"]
        )

        print("Training Nearest Neighbors model...")
        print(df.iloc[0]["title"])

        nn_model = NearestNeighbors(
            n_neighbors=11,
            metric="cosine"
        )

        nn_model.fit(tfidf_matrix)

        # Save model
        with open(
            f"{self.model_dir}/nn_model.pkl",
            "wb"
        ) as f:
            pickle.dump(nn_model, f)

        # Save vectorizer
        with open(
            f"{self.model_dir}/vectorizer.pkl",
            "wb"
        ) as f:
            pickle.dump(vectorizer, f)

        # Save matrix
        with open(
            f"{self.model_dir}/tfidf_matrix.pkl",
            "wb"
        ) as f:
            pickle.dump(tfidf_matrix, f)

        logger.info("Training completed")

        return nn_model
    
   