import os
import time
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

def run_train_model():

    # -------------------------------------------------
    # Logging Setup
    # -------------------------------------------------
    LOG_DIR = "logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("invoice_training")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        LOG_DIR + "training.log", maxBytes=5_000_000, backupCount=3
    )
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    DATA_FILE = "data/db/df_training_corrected.csv"
    SPLIT_DIR = "data/splits/"
    MODEL_DIR = "models/"

    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


    # -------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------
    start = time.time()
    logger.info("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    # label: 1 = correct (saving_loss = 0), 0 = not correct
    df["label_num"] = df["saving_loss"].apply(lambda x: 1 if x == 0 else 0)

    logger.info(f"Loaded {len(df):,} rows in {time.time() - start:.2f}s")


    # -------------------------------------------------
    # 2. Group-Based Train/Val/Test Split (no leakage)
    # -------------------------------------------------
    logger.info("Creating group-based train/val/test splits (based on file_name)...")

    groups = df["file_name"].unique()
    logger.info(f"Found {len(groups):,} unique invoice files.")

    # Split into train + temp
    train_groups, temp_groups = train_test_split(
        groups, test_size=0.20, random_state=42
    )

    # Split temp → val + test
    val_groups, test_groups = train_test_split(
        temp_groups, test_size=0.50, random_state=42
    )

    logger.info(f"Train groups: {len(train_groups):,}")
    logger.info(f"Val groups:   {len(val_groups):,}")
    logger.info(f"Test groups:  {len(test_groups):,}")

    train_df = df[df["file_name"].isin(train_groups)]
    val_df   = df[df["file_name"].isin(val_groups)]
    test_df  = df[df["file_name"].isin(test_groups)]

    logger.info(f"Train rows: {len(train_df):,}")
    logger.info(f"Val rows:   {len(val_df):,}")
    logger.info(f"Test rows:  {len(test_df):,}")

    train_df.to_csv(SPLIT_DIR + "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR + "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR + "test.csv", index=False)

    logger.info("Group-based splits saved successfully.")


    # -------------------------------------------------
    # 3. Load Embedding Model
    # -------------------------------------------------
    start = time.time()
    logger.info("Loading SentenceTransformer model...")

    embedder_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = SentenceTransformer(embedder_name)

    with open(MODEL_DIR + "embedder_name.txt", "w") as f:
        f.write(embedder_name)

    logger.info(f"Embedder loaded in {time.time() - start:.2f}s")


    # -------------------------------------------------
    # 4. Embed Descriptions
    # -------------------------------------------------
    def embed(df, label_name):
        logger.info(f"Embedding {label_name} descriptions ({len(df):,} rows)...")
        start = time.time()
        embeddings = embedder.encode(
            df["description"].astype(str).tolist(),
            show_progress_bar=True
        )
        logger.info(f"Embedded {label_name} in {time.time() - start:.2f}s")
        return embeddings

    X_train_text = embed(train_df, "train")
    X_val_text = embed(val_df, "val")
    X_test_text = embed(test_df, "test")


    # -------------------------------------------------
    # 5. Numeric Features + Scaling
    # -------------------------------------------------
    numeric_cols = ["quantity", "unit_price", "amount",
                    "subtotal", "discount", "taxes", "total_amount"]

    logger.info("Scaling numeric features...")
    start = time.time()

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numeric_cols])
    X_val_num   = scaler.transform(val_df[numeric_cols])
    X_test_num  = scaler.transform(test_df[numeric_cols])

    joblib.dump(scaler, MODEL_DIR + "scaler.pkl")
    logger.info(f"Scaler saved. Scaling done in {time.time() - start:.2f}s")


    # -------------------------------------------------
    # 6. Combine Embeddings + Numeric Features
    # -------------------------------------------------
    logger.info("Combining embeddings with numeric features...")

    X_train = np.hstack([X_train_text, X_train_num])
    X_val   = np.hstack([X_val_text, X_val_num])
    X_test  = np.hstack([X_test_text, X_test_num])

    y_train = train_df["label_num"].values
    y_val   = val_df["label_num"].values
    y_test  = test_df["label_num"].values

    logger.info("Feature combination complete.")


    # -------------------------------------------------
    # 7. Train XGBoost Model
    # -------------------------------------------------
    logger.info("Training XGBoost model...")
    start = time.time()

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.2,
        subsample=0.5,
        colsample_bytree=0.5,
        eval_metric="logloss",
        tree_method="hist"
    )


    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    logger.info(f"Training completed in {time.time() - start:.2f}s")


    # -------------------------------------------------
    # 8. Save Model
    # -------------------------------------------------
    model.save_model(MODEL_DIR + "model_xgb.json")
    logger.info("Model saved to models/model_xgb.json")

    logger.info("Training pipeline finished successfully.")
    # ======================================================

# -------------------------------------------------
if __name__ == "__main__":
    run_train_model()
# -------------------------------------------------