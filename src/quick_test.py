from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineering
from train import ModelTrainer
from recommender import Recommender
import pandas as pd


df = DataIngestion().ingest_data()

print(type(df))
print(df.shape)

df = FeatureEngineering().create_combined_text(df)

trainer = ModelTrainer()

trainer.train(df)

rec = Recommender()

product = df.iloc[0]["title"]

print("Query Product:")
print(product)

print()

print("Recommendations:")

results = rec.recommend(
    product,
    top_k=10
)

for r in results:
    print(r)