import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.logger import logger


class Evaluator:
    """
    Evaluates recommendation quality across multiple dimensions:
    - Intra-List Diversity (ILD): Pairwise diversity of recommended items
    - Mean Similarity: Average cosine similarity to the seed/query item
    - Average Rating: Quality rating of recommended items
    - Category Coverage: Breadth of categories in recommendations
    """

    def evaluate(
        self,
        recommendations: List[Dict[str, Any]],
        query_vector: Optional[Any] = None,
        rec_vectors: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Computes recommendation quality metrics for a given recommendation list.
        """
        if not recommendations:
            return {
                "recommendation_count": 0,
                "intra_list_diversity": 0.0,
                "average_rating": 0.0,
                "mean_similarity": 0.0,
                "category_count": 0,
            }

        rec_count = len(recommendations)

        # 1. Average Rating of recommendations
        ratings = [
            r.get("average_rating", 0.0)
            for r in recommendations
            if r.get("average_rating") is not None
        ]
        avg_rating = round(float(np.mean(ratings)), 2) if ratings else 0.0

        # 2. Category Diversity
        categories = set()
        for r in recommendations:
            cat = r.get("categories")
            if isinstance(cat, list):
                categories.update(cat)
            elif isinstance(cat, str) and cat:
                categories.add(cat)

        # 3. Mean Similarity (if similarity_score provided in recommendations)
        scores = [
            r.get("similarity_score", 0.0)
            for r in recommendations
            if "similarity_score" in r
        ]
        mean_sim = round(float(np.mean(scores)), 4) if scores else 0.0

        # 4. Intra-List Diversity (ILD) using TF-IDF vectors if provided
        ild = 0.0
        if rec_vectors is not None and rec_vectors.shape[0] > 1:
            sim_matrix = cosine_similarity(rec_vectors)
            n = sim_matrix.shape[0]
            # Average off-diagonal distance = (1 - similarity)
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    distances.append(1.0 - sim_matrix[i, j])
            ild = round(float(np.mean(distances)), 4) if distances else 0.0
        elif scores and len(scores) > 1:
            # Fallback ILD proxy from similarity score variance
            ild = round(float(np.std(scores)), 4)

        metrics = {
            "recommendation_count": rec_count,
            "average_rating": avg_rating,
            "mean_similarity": mean_sim,
            "intra_list_diversity": ild,
            "unique_categories_covered": len(categories),
        }
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics