import os
import sqlite3
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler

# ================================
#  LOGGING SETUP
# ================================
LOG_DIR = "logs/"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("db_loader")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Rotating file handler (max 5MB per file)
file_handler = RotatingFileHandler(
    LOG_DIR + "db_load.log", maxBytes=5_000_000, backupCount=3
)
file_handler.setLevel(logging.INFO)

# Log format
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] - %(message)s",
    "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ================================
#  PATHS
# ================================
DB_FILE = "data/db/invoices_data.db"
CSV_FILE = "data/db/df_invoices_cleaned.csv"

os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

# ================================
#  DATABASE FUNCTIONS WITH LOGGING
# ================================
def create_SQLite_db():
    logger.info(f"Connecting to SQLite database → {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    logger.info("Database connection established.")
    return conn, cursor


def create_tables(cursor):
    logger.info("Creating tables (if not exist)...")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        address TEXT,
        email TEXT,
        phone TEXT,
        website TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vendors (
        vendor_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        address TEXT,
        gstin TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS invoices (
        invoice_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        invoice_number TEXT,
        invoice_date TEXT,
        due_date TEXT,
        vendor_id INTEGER,
        customer_id INTEGER,
        subtotal REAL,
        discount REAL,
        taxes REAL,
        total_amount REAL,
        bank_name TEXT,
        bank_address TEXT,
        bank_city TEXT,
        bank_state TEXT,
        bank_zip TEXT,
        bank_country TEXT,
        notes TEXT,
        terms TEXT,
        FOREIGN KEY(vendor_id) REFERENCES vendors(vendor_id),
        FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS line_items (
        line_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_id INTEGER,
        description TEXT,
        quantity REAL,
        unit_price REAL,
        amount REAL,
        FOREIGN KEY(invoice_id) REFERENCES invoices(invoice_id)
    )
    """)

    logger.info("All tables created or already exist.")


def load_data_to_db(cursor, conn):
    logger.info(f"Loading CSV data → {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    logger.info(f"Loaded {len(df):,} rows from CSV.")

    # Insert vendors
    vendor_map = {}
    logger.info("Inserting vendor records...")
    for _, row in df.iterrows():
        vname = row.get("vendor_name")
        if vname not in vendor_map:
            cursor.execute("""
                INSERT INTO vendors (name, address, gstin)
                VALUES (?, ?, ?)
            """, (vname, row.get("vendor_address"), row.get("vendor_gstin")))
            vendor_map[vname] = cursor.lastrowid

    logger.info(f"Inserted {len(vendor_map):,} unique vendors.")

    # Insert customers
    customer_map = {}
    logger.info("Inserting customer records...")
    for _, row in df.iterrows():
        key = (row.get("buyer_name"), row.get("buyer_address"))
        if key not in customer_map:
            cursor.execute("""
                INSERT INTO customers (name, address, email, phone, website)
                VALUES (?, ?, ?, ?, ?)
            """, (
                row.get("buyer_name"),
                row.get("buyer_address"),
                row.get("buyer_email"),
                row.get("buyer_phone"),
                row.get("buyer_website")
            ))
            customer_map[key] = cursor.lastrowid

    logger.info(f"Inserted {len(customer_map):,} unique customers.")

    # Insert invoices
    invoice_map = {}
    logger.info("Inserting invoices...")
    for _, row in df.iterrows():
        inv_num = row.get("invoice_number")

        if inv_num not in invoice_map:
            cursor.execute("""
                INSERT INTO invoices (
                    file_name, invoice_number, invoice_date, due_date,
                    vendor_id, customer_id,
                    subtotal, discount, taxes, total_amount,
                    bank_name, bank_address, bank_city, bank_state, bank_zip, bank_country,
                    notes, terms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get("file_name"),
                row.get("invoice_number"),
                row.get("invoice_date"),
                row.get("due_date"),
                vendor_map.get(row.get("vendor_name")),
                customer_map.get((row.get("buyer_name"), row.get("buyer_address"))),
                row.get("totals_subtotal"),
                row.get("totals_discount"),
                row.get("totals_taxes"),
                row.get("totals_total_amount"),
                row.get("bank_name"),
                row.get("bank_address"),
                row.get("bank_city"),
                row.get("bank_state"),
                row.get("bank_zip"),
                row.get("bank_country"),
                row.get("notes"),
                row.get("terms")
            ))

            invoice_map[inv_num] = cursor.lastrowid

    logger.info(f"Inserted {len(invoice_map):,} invoices.")

    # Insert line items
    logger.info("Inserting line items...")
    line_count = 0
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO line_items (invoice_id, description, quantity, unit_price, amount)
            VALUES (?, ?, ?, ?, ?)
        """, (
            invoice_map.get(row.get("invoice_number")),
            row.get("description"),
            row.get("quantity"),
            row.get("unit_price"),
            row.get("amount")
        ))
        line_count += 1

    logger.info(f"Inserted {line_count:,} line items.")

    conn.commit()
    logger.info("All data committed to database.")


# ================================
#  run_store_data
# ================================
def run_store_data():
    logger.info("Starting database build process...")

    conn, cursor = create_SQLite_db()
    create_tables(cursor)
    load_data_to_db(cursor, conn)
    conn.close()

    logger.info(f"Database successfully built → {DB_FILE}")
    logger.info("Data loading process completed.")


if __name__ == "__main__":
    main()
# ================================