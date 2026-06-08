from fastapi import FastAPI
from src.recommender import Recommender

app = FastAPI(
    title="Amazon Appliance Recommender",
    version="1.0"
)

recommender = Recommender()


@app.get("/")
def home():

    return {
        "message":
        "Amazon Recommendation API Running"
    }

@app.get("/search")
def search(query: str):

    matches = recommender.df[
        recommender.df["title"].str.contains(
            query,
            case=False,
            na=False,
            regex=False
        )
    ]

    return {
        "count": len(matches),
        "products": matches["title"].head(20).tolist()
    }

@app.get("/recommend")
def recommend(
    product_title: str,
    top_k: int = 10
):

    recommendations = recommender.recommend(
        product_title,
        top_k
    )

    return {
        "query": product_title,
        "recommendations": recommendations
    }