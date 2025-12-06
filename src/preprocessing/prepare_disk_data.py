import os
import random
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# ======================================================
# LOGGING SETUP
# ======================================================
LOG_DIR = "logs/"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("save_dataset_to_disk")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
file_handler = RotatingFileHandler(
    LOG_DIR + "save_dataset_to_disk.log", maxBytes=5_000_000, backupCount=3
)

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] - %(message)s",
    "%Y-%m-%d %H:%M:%S"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ======================================================
# MAIN FUNCTION
# ======================================================

def run_saving_data_to_disk():

    logger.info("Starting dataset extraction to disk (1% sample)...")

    # ------------------------------------------------------
    # Configure folders
    # ------------------------------------------------------
    png_dir = Path("data/raw/png")
    pdf_dir = Path("data/raw/pdfs")

    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"PNG output directory: {png_dir}")
    logger.info(f"PDF output directory: {pdf_dir}")

    # ------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------
    logger.info("Loading dataset: mathieu1256/FATURA2-invoices (train split)")

    try:
        dataset = load_dataset("mathieu1256/FATURA2-invoices", split="train")
        n_total = len(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Dataset loaded successfully with {n_total:,} samples")

    # ------------------------------------------------------
    # Take a 1% random sample
    # ------------------------------------------------------
    n_sample = max(1, int(n_total * 0.01))
    indices = random.sample(range(n_total), n_sample)

    logger.info(f"Sampling 1% → {n_sample:,} images")

    dataset_1percent = dataset.select(indices)

    # ------------------------------------------------------
    # Save sampled images as PNG
    # ------------------------------------------------------
    logger.info("Saving sampled images as PNG files...")

    for i, example in enumerate(tqdm(dataset_1percent, desc="Saving PNG files")):
        try:
            image = example["image"]
            image_path = png_dir / f"{i}.png"
            image.save(image_path)
        except Exception as e:
            logger.error(f"Failed to save PNG {i}.png: {e}")

    logger.info(f"Finished saving {n_sample:,} PNG images.")

    # ------------------------------------------------------
    # Shuffle & split 50/50
    # ------------------------------------------------------
    logger.info("Shuffling and splitting into 50% PNG / 50% PDF sets...")

    image_files = sorted(list(png_dir.glob("*.png")))
    random.shuffle(image_files)
    n_half = len(image_files) // 2

    first_half = image_files[:n_half]
    second_half = image_files[n_half:]

    logger.info(f"Images for PDF conversion: {len(first_half):,}")
    logger.info(f"Images to remain as PNG:    {len(second_half):,}")

    # ------------------------------------------------------
    # Convert first half to PDF
    # ------------------------------------------------------
    logger.info("Converting first 50% of images to PDF...")

    for img_file in tqdm(first_half, desc="Converting to PDF"):
        try:
            with Image.open(img_file) as img:
                pdf_path = pdf_dir / f"{img_file.stem}.pdf"
                img.convert("RGB").save(pdf_path)

            img_file.unlink()  # delete PNG
        except Exception as e:
            logger.error(f"Failed to convert {img_file.name} to PDF: {e}")

    logger.info("PDF conversion completed.")

    # ------------------------------------------------------
    # Completion summary
    # ------------------------------------------------------
    remaining_png = list(png_dir.glob("*.png"))

    logger.info("Processing complete!")
    logger.info(f"PDF output folder: {pdf_dir}")
    logger.info(f"Remaining PNG files: {len(remaining_png):,}")
    logger.info("Dataset saved successfully to disk.")

# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    run_saving_data_to_disk()
# -------------------------------------------------