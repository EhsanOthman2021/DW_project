import logging

# Import pipeline runners
from src.preprocessing.main import run as run_preprocessing
from src.db.main import run as run_db
from src.ml.main import run as run_ml

# Configure logging for the entire pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    logging.info("Starting Full Data & ML Pipeline")

    # 1️ PREPROCESSING
    logging.info("Step 1: Running preprocessing pipeline...")
    run_preprocessing()
    logging.info("Preprocessing completed.\n")

    # 2️ DB PIPELINE
    logging.info("Step 2: Running DB (JSON → DF → Clean → Store) pipeline...")
    run_db()
    logging.info("DB pipeline completed.\n")

    # 3️ MACHINE LEARNING PIPELINE
    logging.info("Step 3: Running ML pipeline (prepare → train → evaluate)...")
    run_ml()
    logging.info("ML pipeline completed.\n")

    logging.info("ALL PIPELINES COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
