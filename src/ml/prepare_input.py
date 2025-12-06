import os
import logging
from logging.handlers import RotatingFileHandler
import sqlite3
import pandas as pd
import numpy as np

def run_prepare_input():
    # -------------------------------------------------
    # Logging Setup
    # -------------------------------------------------
    LOG_DIR = "logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("data_preparation")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        LOG_DIR + "data_preparation.log", maxBytes=5_000_000, backupCount=3
    )
    file_handler.setLevel(logging.INFO)

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
    DB_FILE = "data/db/invoices_data.db"
    OUTPUT_CSV = "data/db/df_training_corrected.csv"

    CORRECT_RATIO = 0.60

    # -------------------------------------------------
    # 1. Load data
    # -------------------------------------------------
    logger.info("Loading invoices and line items from SQLite...")

    conn = sqlite3.connect(DB_FILE)
    query = """
    SELECT 
        li.line_item_id,
        li.invoice_id,
        i.invoice_number,
        li.description,
        li.quantity,
        li.unit_price,
        li.amount,
        i.file_name,
        i.subtotal,
        i.discount,
        i.taxes,
        i.total_amount
    FROM line_items li
    JOIN invoices i ON li.invoice_id = i.invoice_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df):,} rows.")

    # -------------------------------------------------
    # 2. Clean NaNs
    # -------------------------------------------------
    logger.info("Cleaning NaN values...")

    numeric_cols = ["quantity", "unit_price", "amount",
                    "subtotal", "discount", "taxes", "total_amount"]
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[['file_name', 'invoice_number', 'description']] = df[['file_name', 'invoice_number', 'description']].fillna("")

    # -------------------------------------------------
    # 3. Compute corrected totals
    # -------------------------------------------------
    logger.info("Computing corrected_total_amount...")

    df['corrected_total_amount'] = (
        df['subtotal'] - df['discount'] + df['taxes']
    ).round(2)

    # -------------------------------------------------
    # 4. Identify incorrect invoices
    # -------------------------------------------------
    logger.info("Identifying invoices with incorrect totals...")

    invoice_totals = (
        df.groupby("file_name")[['total_amount', 'corrected_total_amount']]
        .first()
        .reset_index()
    )

    invoice_totals['is_wrong'] = (
        round(invoice_totals['total_amount'], 2)
        != round(invoice_totals['corrected_total_amount'], 2)
    )

    wrong_invoices = invoice_totals[invoice_totals['is_wrong']]['file_name']

    logger.info(f"Found {len(wrong_invoices):,} incorrect invoices.")

    # -------------------------------------------------
    # 5. Choose % to correct
    # -------------------------------------------------
    num_to_fix = int(len(wrong_invoices) * CORRECT_RATIO)
    logger.info(f"Correcting {num_to_fix:,} invoices ({CORRECT_RATIO*100:.0f}%).")

    invoices_to_fix = wrong_invoices.sample(num_to_fix, random_state=42)

    # -------------------------------------------------
    # 6. Apply corrections
    # -------------------------------------------------
    logger.info("Applying corrected totals...")

    df.loc[df['file_name'].isin(invoices_to_fix), 'total_amount'] = (
        df.loc[df['file_name'].isin(invoices_to_fix), 'corrected_total_amount']
    )

    df['total_formula_correct'] = (
        round(df['total_amount'], 2) ==
        round(df['corrected_total_amount'], 2)
    )

    # -------------------------------------------------
    # 7. Saving (Financial Loss) Calculation
    # -------------------------------------------------
    logger.info("Calculating financial saving/loss per invoice...")

    # Compute difference
    df['saving_loss'] = df['corrected_total_amount'] - df['total_amount']

    # We only care when company LOSES money (corrected_total_amount > total_amount)
    df['saving_loss'] = df['saving_loss'].apply(lambda x: x if x > 0 else 0)

    # Invoice-level saving/loss
    invoice_saving = df.groupby("file_name")['saving_loss'].max().reset_index()

    total_loss = invoice_saving['saving_loss'].sum()

    logger.info(f"Total Financial Loss Across Dataset: {total_loss:.2f}")
    logger.info(f"Average Loss Per Wrong Invoice: {invoice_saving['saving_loss'].mean():.2f}")

    # Merge saving back to df
    df = df.merge(invoice_saving, on="file_name", suffixes=("", "_invoice"))

    # -------------------------------------------------
    # 8. Assign labels
    # -------------------------------------------------
    logger.info("Assigning invoice-level labels...")

    invoice_labels = (
        df.groupby("file_name")['total_formula_correct']
        .all()
        .reset_index()
        .rename(columns={'total_formula_correct': 'invoice_label'})
    )

    invoice_labels['invoice_label'] = invoice_labels['invoice_label'].map(
        {True: "correct", False: "not_correct"}
    )

    df = df.merge(invoice_labels, on="file_name", how="left")
    df['label'] = df['invoice_label']

    # -------------------------------------------------
    # 9. Save dataset
    # -------------------------------------------------
    final_df = df[[
        'file_name', 'invoice_number', 'line_item_id', 'description',
        'quantity', 'unit_price', 'amount',
        'subtotal', 'discount', 'taxes', 'total_amount',
        'corrected_total_amount', 'saving_loss',
        'invoice_label', 'label'
    ]]

    final_df.to_csv(OUTPUT_CSV, index=False)

    logger.info(f"Saved final training dataset to {OUTPUT_CSV}")
    logger.info("Data preparation completed successfully.")

# -------------------------------------------------
if __name__ == "__main__":
    run_prepare_input() 
# -------------------------------------------------