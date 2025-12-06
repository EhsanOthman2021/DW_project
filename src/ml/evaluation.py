import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
import boto3
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import xgboost as xgb

def run_evaluation():
        
    # =====================================================
    # CONFIG
    # =====================================================
    MODEL_DIR = "models/"
    SPLIT_DIR = "data/splits/"
    REPORT_DIR = "reports/"

    S3_BUCKET = "my-fatura2-bucket"              # <-- CHANGE THIS
    S3_PREFIX = "invoice_model_reports/"        # folder in S3

    os.makedirs(REPORT_DIR, exist_ok=True)


    # =====================================================
    # LOGGING
    # =====================================================
    logging.basicConfig(
        filename=os.path.join(REPORT_DIR, "evaluation.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    def trace(msg: str):
        logger.info(msg)
        print(msg)


    trace("==== Evaluation Script Started ====")


    # =====================================================
    # S3 upload helper
    # =====================================================
    s3 = boto3.client("s3")

    def upload_to_s3(local_path, bucket, key):
        if not os.path.exists(local_path):
            trace(f"Skipping missing file: {local_path}")
            return
        try:
            s3.upload_file(local_path, bucket, key)
            trace(f"Uploaded: s3://{bucket}/{key}")
        except Exception as e:
            trace(f"ERROR uploading {local_path} -> {e}")


    # =====================================================
    # LOAD TEST DATA
    # =====================================================
    trace("Loading test dataset...")
    test_df = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))
    y_test = test_df["label_num"].values


    # =====================================================
    # LOAD MODEL
    # =====================================================
    trace("Loading trained model...")
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "model_xgb.json"))


    # =====================================================
    # LOAD EMBEDDER + SCALER
    # =====================================================
    trace("Loading embedder + scaler...")

    with open(os.path.join(MODEL_DIR, "embedder_name.txt"), "r") as f:
        embedder_name = f.read().strip()

    embedder = SentenceTransformer(embedder_name)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


    # =====================================================
    # EMBED TEXT
    # =====================================================
    trace("Embedding text descriptions...")
    X_test_text = embedder.encode(
        test_df["description"].astype(str).tolist(),
        show_progress_bar=True
    )


    # =====================================================
    # PROCESS NUMERIC FEATURES
    # =====================================================
    numeric_cols = [
        "quantity", "unit_price", "amount",
        "subtotal", "discount", "taxes", "total_amount"
    ]

    trace("Scaling numeric columns...")
    X_test_num = scaler.transform(test_df[numeric_cols])

    X_test = np.hstack([X_test_text, X_test_num])


    # =====================================================
    # PREDICTIONS
    # =====================================================
    trace("Running predictions...")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    trace(f"Accuracy = {accuracy:.4f}")
    trace(report)


    # =====================================================
    # ADD PREDICTIONS TO test_df
    # =====================================================
    test_df["pred_num"] = preds
    test_df["pred_prob"] = probs
    test_df["pred_label"] = np.where(preds == 1, "correct", "not_correct")
    test_df["actual_label"] = np.where(test_df["label_num"] == 1, "correct", "not_correct")


    # =====================================================
    # SAVE UPDATED test.csv
    # =====================================================
    updated_csv = os.path.join(REPORT_DIR, "test_with_predictions.csv")
    test_df.to_csv(updated_csv, index=False)
    trace(f"Saved updated test CSV: {updated_csv}")


    # =====================================================
    # PLOTS
    # =====================================================
    trace("Creating evaluation plots...")

    # --- Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    fig_cm = px.imshow(
        cm,
        x=["not_correct", "correct"],
        y=["not_correct", "correct"],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    fig_cm.write_html(os.path.join(REPORT_DIR, "confusion_matrix.html"))

    # --- ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                line=dict(dash="dash"), name="Random"))
    fig_roc.update_layout(title="ROC Curve")
    fig_roc.write_html(os.path.join(REPORT_DIR, "roc_curve.html"))

    # --- Precision Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines"))
    fig_pr.update_layout(title="Precision-Recall Curve")
    fig_pr.write_html(os.path.join(REPORT_DIR, "precision_recall_curve.html"))

    # --- Accuracy Gauge
    fig_acc = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy * 100,
        title={"text": "Accuracy (%)"},
        gauge={"axis": {"range": [0, 100]}}
    ))
    fig_acc.write_html(os.path.join(REPORT_DIR, "accuracy.html"))

    trace("Plots created successfully.")


    # =====================================================
    # UPLOAD TO S3
    # =====================================================
    trace("Uploading reports to S3...")

    files = [
        "test_with_predictions.csv",
        "confusion_matrix.html",
        "roc_curve.html",
        "precision_recall_curve.html",
        "accuracy.html",
        "evaluation.log",
    ]

    for f in files:
        upload_to_s3(
            os.path.join(REPORT_DIR, f),
            S3_BUCKET,
            f"{S3_PREFIX}{f}"
        )

    # Metrics JSON
    metrics_json = os.path.join(REPORT_DIR, "metrics_summary.json")
    pd.Series({"accuracy": float(accuracy)}).to_json(metrics_json, indent=4)
    upload_to_s3(metrics_json, S3_BUCKET, f"{S3_PREFIX}metrics_summary.json")

    trace("==== Evaluation Completed Successfully ====")
    # =====================================================

# -------------------------------------------------
if __name__ == "__main__":
    run_evaluation()
# -------------------------------------------------