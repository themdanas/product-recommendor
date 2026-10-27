import os
import re
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.logger import logger
from src.feature_engineering import FeatureEngineering


class Recommender:
    """
    Production recommendation engine supporting both Item-to-Item
    similarity and Semantic Free-Text query matching.
    """

    PLACEHOLDER_IMAGE = "https://images.unsplash.com/photo-1584269600464-37b1b58a9fe7?auto=format&fit=crop&w=600&q=80"

    def __init__(
        self,
        data_path: Optional[str] = None,
        model_dir: str = "artifacts/models",
    ):
        self.model_dir = model_dir
        self.fe = FeatureEngineering()

        # 1. Resolve dataset path (parquet preferred over csv)
        if data_path:
            self.data_path = data_path
        elif os.path.exists("artifacts/data/processed_data.parquet"):
            self.data_path = "artifacts/data/processed_data.parquet"
        elif os.path.exists("artifacts/data/processed_data.csv"):
            self.data_path = "artifacts/data/processed_data.csv"
        else:
            raise FileNotFoundError(
                "Could not find processed dataset at 'artifacts/data/processed_data.parquet' or '.csv'. "
                "Run 'python main.py --pipeline' first."
            )

        logger.info(f"Recommender loading catalog from {self.data_path}...")
        if self.data_path.endswith(".parquet"):
            self.df = pd.read_parquet(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path, low_memory=False)

        # 2. Load model artifacts
        nn_path = os.path.join(self.model_dir, "nn_model.pkl")
        vec_path = os.path.join(self.model_dir, "vectorizer.pkl")
        mat_path = os.path.join(self.model_dir, "tfidf_matrix.pkl")

        for p in [nn_path, vec_path, mat_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Model artifact not found at {p}. Run 'python main.py --pipeline' first."
                )

        logger.info("Loading model, vectorizer, and TF-IDF matrix...")
        with open(nn_path, "rb") as f:
            self.model = pickle.load(f)
        with open(vec_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(mat_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        # 3. Build fast ASIN index map
        self.asin_to_index = {}
        if "parent_asin" in self.df.columns:
            for idx, asin in enumerate(self.df["parent_asin"]):
                if pd.notna(asin):
                    self.asin_to_index[str(asin).strip()] = idx

        logger.info(f"Recommender initialized with {len(self.df)} products.")

    def _extract_image_url(self, val: Any) -> str:
        """Extracts the first Amazon media image URL from raw string/dict."""
        if val is None or pd.isna(val):
            return self.PLACEHOLDER_IMAGE
        text = str(val)
        matches = re.findall(r"https://m\.media-amazon\.com/images/I/[^\"\'\s\)]+", text)
        if matches:
            return matches[0]
        # Any fallback http URL
        urls = re.findall(r"https?://[^\s\"\'<>]+", text)
        return urls[0] if urls else self.PLACEHOLDER_IMAGE

    def _parse_categories(self, val: Any) -> List[str]:
        """Parses categories array or string into clean string list."""
        if val is None or pd.isna(val):
            return ["Appliances"]
        if isinstance(val, (list, tuple)):
            return [str(c).strip() for c in val if c]
        # Extract quoted segments
        cats = re.findall(r"'([^']+)'", str(val))
        return cats if cats else ["Appliances"]

    def _format_product(self, idx: int, similarity_score: Optional[float] = None) -> Dict[str, Any]:
        """Converts a row index in self.df to a rich product metadata dictionary."""
        row = self.df.iloc[idx]
        price_val = row.get("price")
        if pd.isna(price_val) or str(price_val).lower() in ("none", "nan", ""):
            formatted_price = "$ --"
        else:
            try:
                # If numeric or contains digits
                formatted_price = f"${float(str(price_val).replace('$', '').strip()):.2f}"
            except Exception:
                formatted_price = str(price_val)

        rating = row.get("average_rating", 0.0)
        rating = float(rating) if pd.notna(rating) else 0.0

        rating_num = row.get("rating_number", 0)
        rating_num = int(rating_num) if pd.notna(rating_num) else 0

        product = {
            "index": int(idx),
            "parent_asin": str(row.get("parent_asin", "")),
            "title": str(row.get("title", "Unknown Product")),
            "average_rating": round(rating, 2),
            "rating_number": rating_num,
            "price": formatted_price,
            "image_url": self._extract_image_url(row.get("images")),
            "categories": self._parse_categories(row.get("categories")),
            "store": str(row.get("store", "Amazon")),
        }
        if similarity_score is not None:
            product["similarity_score"] = round(float(similarity_score), 4)
            product["match_percentage"] = f"{int(max(0.0, min(1.0, similarity_score)) * 100)}%"

        return product

    def recommend(self, product_identifier: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Recommends products similar to an existing product, identified
        by parent_asin or product title substring.
        """
        if not product_identifier or not product_identifier.strip():
            return []

        clean_id = product_identifier.strip()

        # 1. Check exact ASIN match
        if clean_id in self.asin_to_index:
            seed_idx = self.asin_to_index[clean_id]
        else:
            # 2. Substring search in title
            matches = self.df[
                self.df["title"].str.contains(
                    re.escape(clean_id),
                    case=False,
                    na=False,
                )
            ]
            if matches.empty:
                logger.warning(f"No direct title/ASIN match for: '{clean_id}'. Falling back to query vectorization.")
                return self.recommend_by_query(clean_id, top_k=top_k)
            seed_idx = matches.index[0]

        # 3. Query Nearest Neighbors
        seed_vector = self.tfidf_matrix[seed_idx]
        distances, indices = self.model.kneighbors(
            seed_vector,
            n_neighbors=min(top_k + 1, len(self.df)),
        )

        results = []
        for dist, idx in zip(distances[0][1:], indices[0][1:]):
            sim_score = 1.0 - dist
            results.append(self._format_product(idx, similarity_score=sim_score))

        return results

    def recommend_by_query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Semantic Search: Vectorizes arbitrary user query and returns nearest products.
        """
        if not query_text or not query_text.strip():
            return []

        cleaned_query = self.fe.clean_text(query_text)
        query_vector = self.vectorizer.transform([cleaned_query])

        distances, indices = self.model.kneighbors(
            query_vector,
            n_neighbors=min(top_k, len(self.df)),
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            sim_score = 1.0 - dist
            results.append(self._format_product(idx, similarity_score=sim_score))

        return results

    def search_products(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Searches products by title keyword for autocomplete and quick suggestions."""
        if not query or not query.strip():
            return []

        matches = self.df[
            self.df["title"].str.contains(
                re.escape(query.strip()),
                case=False,
                na=False,
            )
        ]
        results = []
        for idx in matches.index[:limit]:
            results.append(self._format_product(idx))
        return results

    def get_product_by_asin(self, asin: str) -> Optional[Dict[str, Any]]:
        """Retrieves full details for a product by its parent_asin."""
        asin_clean = asin.strip()
        if asin_clean in self.asin_to_index:
            idx = self.asin_to_index[asin_clean]
            return self._format_product(idx)
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Returns catalog and model stats."""
        return {
            "total_products": len(self.df),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "average_rating": round(float(self.df["average_rating"].mean()), 2),
            "matrix_shape": list(self.tfidf_matrix.shape),
        }