import pandas as pd


class DataValidation:

    REQUIRED_COLUMNS = [
        "title",
        "description",
        "features",
        "categories",
        "parent_asin",
        "average_rating",
        "rating_number"
    ]

    def validate(self, df: pd.DataFrame):

        # Check columns
        missing_cols = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing columns: {missing_cols}"
            )

        # Unique product ids
        assert df["parent_asin"].duplicated().sum() == 0

        # Ratings range
        assert (
            df["average_rating"]
            .between(0, 5)
            .all(),
            (
                f"Validation Failed! Found ratings outside 0-5 range. "
                f"Min rating: {df['average_rating'].min()}, Max rating: {df['average_rating'].max()}"
            )
        )

        # Review counts
        assert (
            df["rating_number"] >= 0
        ).all()

        return True