import os
import re
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler


def setup_logger():
    """Configure application logger with console + rotating file handlers."""
    LOG_DIR = "logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("clean_invoices")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        LOG_DIR + "clean_invoices.log",
        maxBytes=5_000_000,
        backupCount=3
    )
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def clean_text(text):
    """Normalize whitespace and remove control characters."""
    if pd.isna(text):
        return text
    text = re.sub(r"[\n\r\t]", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run_clean_df():
    logger = setup_logger()

    INPUT_CSV = "data/db/df_invoices.csv"
    OUTPUT_CSV = "data/db/df_invoices_cleaned.csv"

    logger.info(f"Loading CSV → {INPUT_CSV}")

    try:
        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise

    # -------------------------------------------------
    # 1️⃣ Clean text columns
    # -------------------------------------------------
    logger.info("Cleaning text columns...")

    text_columns = [
        "vendor_name", "vendor_address",
        "buyer_name", "buyer_address", "buyer_email", "buyer_phone", "buyer_website",
        "description", "notes", "terms",
        "bank_name", "bank_address", "bank_city", "bank_state", "bank_country",
        "id"
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
        else:
            logger.warning(f"Missing text column: {col}")

    logger.info("Text cleaning complete.")

    # -------------------------------------------------
    # 2️⃣ Clean numeric columns
    # -------------------------------------------------
    logger.info("Cleaning numeric columns...")

    numeric_columns = [
        "quantity", "unit_price", "amount",
        "totals_subtotal", "totals_discount",
        "totals_taxes", "totals_total_amount"
    ]

    for col in numeric_columns:
        if col not in df.columns:
            logger.warning(f"Missing numeric column: {col}")
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^\d\.]", "", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Numeric cleaning complete.")

    # -------------------------------------------------
    # 3️⃣ Standardize dates
    # -------------------------------------------------
    logger.info("Standardizing date formats...")

    date_columns = ["invoice_date", "due_date"]

    for col in date_columns:
        if col not in df.columns:
            logger.warning(f"Missing date column: {col}")
            continue

        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

    logger.info("Date cleaning complete.")

    # -------------------------------------------------
    # 4️⃣ Save cleaned CSV
    # -------------------------------------------------
    logger.info(f"Saving cleaned CSV → {OUTPUT_CSV}")

    try:
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info("Cleaned file saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save cleaned CSV: {e}")
        raise


# Run when executed directly
if __name__ == "__main__":
    run_clean_df()
#-------------------------------------------------