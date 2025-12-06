from src.ml.prepare_input import run_prepare_input 
from src.ml.train_model import run_train_model
from src.ml.evaluation import run_evaluation
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run():
    logging.info("Step 1: Preparing input data")
    run_prepare_input()

    logging.info("Step 2: Training the model")
    run_train_model()

    logging.info("Step 3: Evaluating the model")
    run_evaluation()
    

if __name__ == "__main__":
    run()