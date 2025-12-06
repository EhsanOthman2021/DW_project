from src.db.load_json_to_df import run_load_json_to_df
from src.db.clean_df import run_clean_df
from src.db.store_data import run_store_data
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run():
    logging.info("Step 1: Loading JSON files into a DataFrame")
    run_load_json_to_df()

    logging.info("Step 2: Cleaning the DataFrame")
    run_clean_df()

    logging.info("Step 3: Storing the cleaned data")
    run_store_data()
    

if __name__ == "__main__":
    run()