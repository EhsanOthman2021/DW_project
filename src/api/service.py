import os
import json
import boto3
import pandas as pd
from fastapi import FastAPI

PROCESSED_DIR = "data/processed"

# --------------------------
# S3 Settings
# --------------------------
AWS_BUCKET = "my-fatura2-bucket"
S3_KEY = "invoice_model_reports/test_with_predictions.csv"
CSV_PATH = "test_with_predictions.csv"


# --------------------------
# Download CSV from S3
# --------------------------
def download_csv_from_s3():
    if not os.path.exists(CSV_PATH):
        print("Downloading CSV from S3...")
        s3 = boto3.client("s3")
        s3.download_file(AWS_BUCKET, S3_KEY, CSV_PATH)
        print("CSV downloaded!")
    else:
        print("CSV already exists locally.")

if os.environ.get("DISABLE_S3_DOWNLOAD") != "1":
    download_csv_from_s3()


# --------------------------
# Load CSV
# --------------------------
df = pd.read_csv(CSV_PATH)
df["file_name"] = df["file_name"].astype(str)
df = df.set_index("file_name")


# --------------------------
# Utility Functions
# --------------------------
def clean_filename(name: str):
    """Return only the base name without extension."""
    return os.path.splitext(name)[0].lower()


def find_processed_file(request_name: str):
    base_name = clean_filename(request_name)

    files = os.listdir(PROCESSED_DIR)
    json_files = [f for f in files if f.endswith(".json")]

    # Exact match
    if base_name + ".json" in json_files:
        return os.path.join(PROCESSED_DIR, base_name + ".json")

    # Partial match
    for f in json_files:
        if base_name in clean_filename(f):
            return os.path.join(PROCESSED_DIR, f)

    return None


# --------------------------
# FASTAPI APP
# --------------------------
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Invoice lookup API is running"}


# --------------------------
# PREDICTION ENDPOINT
# --------------------------
@app.get("/predict/{file_name}")
def api_predict(file_name: str):

    # 1️⃣ Find processed JSON file
    file_path = find_processed_file(file_name)
    if not file_path:
        return {"error": f"No processed JSON file found for '{file_name}'"}

    # 2️⃣ Read JSON file
    try:
        with open(file_path, "r", encoding="utf-8") as fp:
            content = json.load(fp)
    except Exception as e:
        return {"error": "Error reading JSON file", "details": str(e)}

    # 3️⃣ Extract JSON base name
    json_base_name = clean_filename(os.path.basename(file_path))

    # 4️⃣ Find CSV row
    matches = df[df.index.str.contains(json_base_name, case=False, regex=False)]

    # 🔥 If JSON exists but no CSV row found
    if matches.empty:
        return {
            "error": "This file was failed to be processed by the model.",
            "file": file_name
        }

    # 5️⃣ Extract labels
    csv_row = matches.iloc[0].to_dict()
    actual_label = csv_row.get("actual_label")
    pred_label = csv_row.get("pred_label")

    # 6️⃣ Build response
    output = content.copy()
    output["actual_label"] = actual_label
    output["predicted_label"] = pred_label

    return output

# --------------------------
# MAIN ENTRY POINT
# --------------------------
def main():
    import uvicorn
    uvicorn.run("src.api.service:app", host="0.0.0.0", port=8080, reload=True)
    # python -m src.api.service
    # http://localhost:8080/predict/31.png
   # ['31.json', '74.json', '77.json', 'test_10.json', 'test_13.json', 'test_4.json']

if __name__ == "__main__":
    main()
