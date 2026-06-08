from src.data_ingestion import DataIngestion


if __name__ == "__main__":

    ingestion = DataIngestion()

    df = ingestion.ingest_data()

    print(df.head())