from src.recommender import Recommender

rec = Recommender()

print("DF Length:", len(rec.df))
print("TFIDF Shape:", rec.tfidf_matrix.shape)

print(rec.recommend("Samsung Refrigerator"))
print(rec.recommend("LG Washing Machine"))