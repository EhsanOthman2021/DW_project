import os
import random
import boto3
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# =====================================
# LOGGING SETUP
# =====================================
LOG_DIR = "logs/"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("s3_upload")
logger.setLevel(logging.INFO)

# Handlers
console_handler = logging.StreamHandler()
file_handler = RotatingFileHandler(
    LOG_DIR + "s3_upload.log", maxBytes=5_000_000, backupCount=3
)

# Format
fmt = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] - %(message)s",
    "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(fmt)
file_handler.setFormatter(fmt)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =====================================
# UPLOAD FUNCTION
# =====================================

def run_saving_data_to_S3():

    logger.info("Starting S3 upload pipeline (1% sample of dataset)...")

    # -------------------------------------
    # AWS S3 CONFIG (Use ENV VARS)
    # -------------------------------------
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_key or not aws_secret:
        logger.error("   AWS credentials missing! Set environment variables:")
        logger.error("   export AWS_ACCESS_KEY_ID=xxx")
        logger.error("   export AWS_SECRET_ACCESS_KEY=xxx")
        return

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name="eu-north-1",
    )

    bucket_name = "my-fatura2-bucket"
    s3_prefix = "png/"

    logger.info(f"S3 Bucket: {bucket_name}")
    logger.info(f"S3 Prefix: {s3_prefix}")

    # -------------------------------------
    # LOAD DATASET
    # -------------------------------------
    logger.info("Loading dataset: mathieu1256/FATURA2-invoices (split=test)")
    dataset = load_dataset("mathieu1256/FATURA2-invoices", split="test")

    n_total = len(dataset)
    n_sample = max(1, int(n_total * 0.01))

    logger.info(f"Total images: {n_total:,}")
    logger.info(f"Sampling 1% → {n_sample:,} images")

    # -------------------------------------
    # SAMPLE RANDOM 1%
    # -------------------------------------
    indices = random.sample(range(n_total), n_sample)
    dataset_1percent = dataset.select(indices)

    logger.info("Random sample selected successfully.")

    # -------------------------------------
    # TEMP LOCAL FOLDER
    # -------------------------------------
    tmp_dir = Path("tmp_test_1percent")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Temporary directory created: {tmp_dir}")

    # -------------------------------------
    # PROCESS EACH IMAGE
    # -------------------------------------
    upload_count = 0

    for i, example in enumerate(tqdm(dataset_1percent, desc="Uploading 1% to S3")):
        try:
            img = example["image"]
            filename = f"test_{i}.png"
            local_path = tmp_dir / filename

            # Save image
            img.save(local_path)

            # Upload to S3
            s3.upload_file(str(local_path), bucket_name, f"{s3_prefix}{filename}")
            upload_count += 1

            logger.info(f"Uploaded: {filename} → S3://{bucket_name}/{s3_prefix}")

            # Remove temp file
            os.remove(local_path)

        except Exception as e:
            logger.error(f"Error uploading {filename}: {e}")

    logger.info(f"✔ Upload complete! {upload_count}/{n_sample} images uploaded successfully.")

    logger.info("Cleaning up temporary folder.")
    try:
        tmp_dir.rmdir()
    except:
        logger.warning("Temporary directory not empty, leaving as-is.")

    logger.info("Pipeline finished.")


# =====================================
# MAIN
# =====================================
if __name__ == "__main__":
    run_saving_data_to_S3()
# -------------------------------------------------