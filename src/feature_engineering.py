import re
import pandas as pd


class FeatureEngineering:

    def clean_text(self, text):

        text = str(text).lower()

        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def create_combined_text(self, df):

        df = df[df["rating_number"] >= 5].copy()        
        df["combined_text"] = (
            df["title"].fillna("").astype(str)
            + " "
            + df["features"].astype(str)
            + " "
            + df["categories"].astype(str)
            + " "
            + df["description"].astype(str)
        )

        df["combined_text"] = df["combined_text"].apply(
            self.clean_text
        )


        return df
    
   
    #this is the change in the file