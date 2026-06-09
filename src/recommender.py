import pickle
import pandas as pd
import numpy as np
from pathlib import Path




class Recommender:

    def __init__(self):
        self.df = pd.read_csv(
            "artifacts/data/processed_data.csv"
        )

        self.model = pickle.load(
            open(
                "artifacts/models/nn_model.pkl",
                "rb"
            )
        )

        self.vectorizer = pickle.load(
            open(
                "artifacts/models/vectorizer.pkl",
                "rb"
            )
        )

        self.tfidf_matrix = pickle.load(
            open(
                "artifacts/models/tfidf_matrix.pkl",
                "rb"
            )
        )
    

    def recommend(self,product_title,top_k=10):

        matches = self.df[
            self.df["title"].str.contains(
                product_title,
                case=False,
                na=False,
                regex=False
            )
        ]

        if matches.empty:
            return print("No matching product found.")
        
        idx = matches.index[0]

        distances, indices = self.model.kneighbors(
            self.tfidf_matrix[idx],
            n_neighbors=top_k + 1

        )

        recommendations = []

        for i in indices[0][1:]:
            recommendations.append(
                self.df.iloc[i]["title"]
            )

        return recommendations
    
    