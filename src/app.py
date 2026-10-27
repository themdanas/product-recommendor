from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from src.recommender import Recommender
from src.logger import logger

app = FastAPI(
    title="Amazon Appliance Recommender API",
    description="High-performance ML recommendation engine and semantic product search for Amazon Appliances.",
    version="2.0.0",
)

# 1. Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Global recommender instance with lazy/safe fallback
try:
    recommender = Recommender()
except Exception as e:
    logger.error(f"Failed to initialize Recommender on app startup: {e}")
    recommender = None


# 3. Pydantic Response Schemas
class ProductItem(BaseModel):
    index: Optional[int] = None
    parent_asin: str
    title: str
    average_rating: float
    rating_number: int
    price: str
    image_url: str
    categories: List[str]
    store: str
    similarity_score: Optional[float] = None
    match_percentage: Optional[str] = None


class RecommendationResponse(BaseModel):
    query: str
    mode: str = "item_similarity"
    count: int
    recommendations: List[ProductItem]


class SearchResponse(BaseModel):
    query: str
    count: int
    products: List[ProductItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    total_products: int
    stats: Dict[str, Any]


# 4. API Endpoints
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Amazon Appliance Recommendation API is Running",
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    if recommender is None:
        return HealthResponse(
            status="degraded",
            version="2.0.0",
            model_loaded=False,
            total_products=0,
            stats={"error": "Recommender models not loaded"},
        )
    stats = recommender.get_stats()
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded=True,
        total_products=stats["total_products"],
        stats=stats,
    )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
def search(
    query: str = Query(..., min_length=1, description="Keywords to search product titles"),
    limit: int = Query(20, ge=1, le=100, description="Max products to return"),
):
    """Searches products by title keyword."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender service not ready")
    results = recommender.search_products(query, limit=limit)
    return SearchResponse(
        query=query,
        count=len(results),
        products=results,
    )


@app.get("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
def recommend(
    product_title: str = Query(..., min_length=1, description="Product title substring or ASIN"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations"),
):
    """
    Recommends products similar to an existing product in the catalog
    matched by title or ASIN.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender service not ready")

    results = recommender.recommend(product_title, top_k=top_k)
    return RecommendationResponse(
        query=product_title,
        mode="item_similarity",
        count=len(results),
        recommendations=results,
    )


@app.get("/recommend/query", response_model=RecommendationResponse, tags=["Recommendations"])
def recommend_by_query(
    query: str = Query(..., min_length=1, description="Natural language search description"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations"),
):
    """
    Semantic recommendation: Vectorizes arbitrary free-text user requirements
    and matches nearest products in TF-IDF cosine vector space.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender service not ready")

    results = recommender.recommend_by_query(query, top_k=top_k)
    return RecommendationResponse(
        query=query,
        mode="semantic_search",
        count=len(results),
        recommendations=results,
    )


@app.get("/product/{asin}", response_model=ProductItem, tags=["Products"])
def get_product(asin: str):
    """Retrieves metadata for a specific product by its parent ASIN."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender service not ready")

    product = recommender.get_product_by_asin(asin)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product with ASIN '{asin}' not found")
    return product