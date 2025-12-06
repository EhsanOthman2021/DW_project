import os
import json
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler

def run_load_json_to_df():
    # -------------------------------------------------
    # Logging Setup
    # -------------------------------------------------
    LOG_DIR = "logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("json_to_csv")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Rotating file handler (5MB max, 3 backups)
    file_handler = RotatingFileHandler(
        LOG_DIR + "json_to_csv.log", maxBytes=5_000_000, backupCount=3
    )
    file_handler.setLevel(logging.INFO)

    # Log formatting
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    folder_path = 'data/processed'
    output_csv = 'data/db/df_invoices.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # -------------------------------------------------
    # Helper: clean currency / numeric values
    # -------------------------------------------------
    def clean_numeric(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return value
        value = value.replace("$", "").replace(",", "").strip()
        try:
            return float(value)
        except:
            return value  # keep original if not convertible


    # -------------------------------------------------
    # Process JSON files
    # -------------------------------------------------
    logger.info("Starting JSON → CSV conversion...")
    logger.info(f"Reading JSON files from: {folder_path}")

    all_rows = []
    file_count = 0
    line_item_count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_count += 1
            file_path = os.path.join(folder_path, filename)

            logger.info(f"Processing file: {filename}")

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read {filename}: {e}")
                continue

            # Extract line items (list)
            line_items = data.get("line_items", [])
            logger.info(f"Found {len(line_items)} line items in {filename}")

            for item in line_items:
                line_item_count += 1

                row = {
                    "file_name": filename,
                    "invoice_id": data.get("id"),  # optional (your sample JSON has no id)
                    "invoice_number": data.get("invoice_number"),
                    "invoice_date": data.get("invoice_date"),
                    "due_date": data.get("due_date"),

                    # Vendor info
                    **{f"vendor_{k}": v for k, v in data.get("vendor", {}).items()},

                    # Buyer info
                    **{f"buyer_{k}": v for k, v in data.get("buyer", {}).items()},

                    # Totals (clean numeric)
                    **{f"totals_{k}": clean_numeric(v) for k, v in data.get("totals", {}).items()},

                    # Bank info
                    **{f"bank_{k}": v for k, v in data.get("bank_info", {}).items()},

                    # Notes and terms
                    "notes": data.get("notes"),
                    "terms": data.get("terms"),

                    # Line item fields (clean numeric where needed)
                    "line_item_id": item.get("id"),
                    "description": item.get("description"),
                    "quantity": clean_numeric(item.get("quantity")),
                    "unit_price": clean_numeric(item.get("unit_price")),
                    "amount": clean_numeric(item.get("amount")),
                }

                all_rows.append(row)

    logger.info(f"Processed {file_count:,} JSON files.")
    logger.info(f"Extracted {line_item_count:,} total line items.")

    # -------------------------------------------------
    # Create DataFrame
    # -------------------------------------------------
    logger.info("Converting data to DataFrame...")

    df = pd.DataFrame(all_rows)
    logger.info(f"Final DataFrame contains {len(df):,} rows and {df.shape[1]} columns.")

    # -------------------------------------------------
    # Save CSV
    # -------------------------------------------------
    df.to_csv(output_csv, index=False)
    logger.info(f"CSV file saved successfully → {output_csv}")
    logger.info("JSON → CSV pipeline completed.")

# -------------------------------------------------
if __name__ == "__main__":
    run_load_json_to_df()
# -------------------------------------------------
