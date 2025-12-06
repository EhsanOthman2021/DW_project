from src.preprocessing.prepare_disk_data import run_saving_data_to_disk 
from src.preprocessing.prepare_S3_data import run_saving_data_to_S3
from src.preprocessing.parse_data import run_parsing_into_structured_json
from src.preprocessing.apply_LLM import run_applying_LLM
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run():
    logging.info("Step 1: Saving data to disk")
    run_saving_data_to_disk()

    logging.info("Step 2: Saving data to S3")
    run_saving_data_to_S3()

    logging.info("Step 3: Parsing raw JSON files into structured JSON")
    run_parsing_into_structured_json()

    logging.info("Step 4: Applying LLM to process files")
    run_applying_LLM()