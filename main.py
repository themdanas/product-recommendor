import argparse
import sys
from src.logger import logger
from src.pipeline import MLPipeline
from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation


def main():
    parser = argparse.ArgumentParser(
        description="E-Commerce Product Recommender CLI"
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run the complete end-to-end ML pipeline",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run data ingestion only",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run data validation on existing dataset",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the FastAPI backend server with uvicorn",
    )
    parser.add_argument(
        "--frontend",
        action="store_true",
        help="Launch the Streamlit frontend UI",
    )

    args = parser.parse_args()

    # Default action if no arguments provided: run pipeline
    if len(sys.argv) == 1 or args.pipeline:
        logger.info("Executing full ML pipeline...")
        pipeline = MLPipeline()
        results = pipeline.run_pipeline()
        logger.info(f"Pipeline executed successfully. Processed {results['product_count']} products.")

    elif args.ingest:
        logger.info("Executing data ingestion...")
        ingestion = DataIngestion()
        df = ingestion.ingest_data()
        logger.info(f"Ingested {len(df)} rows. Sample:\n{df.head(3)}")

    elif args.validate:
        logger.info("Executing data validation...")
        ingestion = DataIngestion()
        df = ingestion.ingest_data()
        validator = DataValidation()
        report = validator.validate(df)
        logger.info(f"Validation summary: {report}")

    elif args.serve:
        import uvicorn
        logger.info("Starting FastAPI backend on http://0.0.0.0:8000...")
        uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)

    elif args.frontend:
        import os
        logger.info("Starting Streamlit frontend...")
        os.system("streamlit run src/frontend.py --server.port 8501")


if __name__ == "__main__":
    main()