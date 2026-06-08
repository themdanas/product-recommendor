from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineering
from train import ModelTrainer
from recommender import Recommender


df = DataIngestion().ingest_data()

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
    df
)

for r in results:
    print(r)