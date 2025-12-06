import os
import json
import logging
from logging.handlers import RotatingFileHandler
import pytesseract
from PIL import Image
import boto3

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def run_parsing_into_structured_json():
    
    # ======================================================
    # PATHS
    # ======================================================
    RAW_PNG_DIR = "data/raw/png"
    RAW_PDF_DIR = "data/raw/pdfs"

    # FIXED S3 SETTINGS
    S3_BUCKET = "my-fatura2-bucket"     # ⬅ your bucket, fixed
    S3_PREFIX = "png/"                  # ⬅ folder for PNG invoices
    S3_PDF_PREFIX = "pdfs/"             # ⬅ folder for PDF invoices (if any)

    OUTPUT_DIR = "processed_pre"
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # ======================================================
    # LOGGING
    # ======================================================
    LOG_DIR = "logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("invoice_raw_text_saver")
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    file = RotatingFileHandler(LOG_DIR + "raw_text.log", maxBytes=5_000_000, backupCount=3)

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s", "%Y-%m-%d %H:%M:%S")
    console.setFormatter(fmt)
    file.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file)


    # ======================================================
    # LOAD LLM MODEL
    # ======================================================
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

    logger.info("Loading Phi-3 model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False
    )

    logger.info("Phi-3 loaded successfully.")


    # ======================================================
    # OCR FUNCTIONS
    # ======================================================
    def ocr_png_local(path):
        img = Image.open(path)
        return pytesseract.image_to_string(img)


    def ocr_png_s3(key):
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        img = Image.open(obj["Body"])
        return pytesseract.image_to_string(img)


    # ===== TEXTRACT PDF OCR (no Poppler needed) =====
    textract = boto3.client("textract")

    def ocr_pdf_s3(key):
        logger.info(f"Textract PDF OCR → {key}")

        response = textract.analyze_document(
            Document={"S3Object": {"Bucket": S3_BUCKET, "Name": key}},
            FeatureTypes=["TEXT_DETECTION"]
        )

        text = ""
        for block in response.get("Blocks", []):
            if block["BlockType"] == "LINE":
                text += block["Text"] + "\n"

        return text


    # ======================================================
    # PROMPT BUILDER
    # ======================================================
    def build_prompt(invoice_text):
        return f"""
    You are an assistant that extracts all invoice data into JSON.
    Extract:
    - Invoice number
    - Invoice date
    - Due date
    - Vendor name & address
    - Buyer name & address, email, phone, website
    - Line items (description, quantity, unit price, amount)
    - Subtotal, discount, taxes, total amount
    - Bank info
    - Notes, terms
    Return strictly in JSON format.

    Invoice Text:
    {invoice_text}

    JSON Output:
    ```json
    """


    # ======================================================
    # LOAD LOCAL INPUTS
    # ======================================================
    def load_local_inputs():
        inputs = []

        # LOCAL PNGs
        if os.path.exists(RAW_PNG_DIR):
            for f in os.listdir(RAW_PNG_DIR):
                if f.lower().endswith(".png"):
                    path = os.path.join(RAW_PNG_DIR, f)
                    text = ocr_png_local(path)
                    inputs.append((f, text))

        # LOCAL PDFs → use Textract if needed
        if os.path.exists(RAW_PDF_DIR):
            for f in os.listdir(RAW_PDF_DIR):
                if f.lower().endswith(".pdf"):
                    logger.warning("Local PDF OCR skipped (no Poppler). Place PDFs in S3 instead.")
        
        return inputs


    # ======================================================
    # LOAD S3 INPUTS (PNG + PDF)
    # ======================================================
    def load_s3_inputs():
        s3 = boto3.client("s3")
        inputs = []

        # ---------- PNG FROM S3 ----------
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        if "Contents" in resp:
            for obj in resp["Contents"]:
                key = obj["Key"]
                if key.lower().endswith(".png"):
                    filename = key.split("/")[-1]
                    text = ocr_png_s3(key)
                    inputs.append((filename, text))

        # ---------- PDF FROM S3 via Textract ----------
        resp_pdf = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PDF_PREFIX)
        if "Contents" in resp_pdf:
            for obj in resp_pdf["Contents"]:
                key = obj["Key"]
                if key.lower().endswith(".pdf"):
                    filename = key.split("/")[-1]
                    text = ocr_pdf_s3(key)
                    inputs.append((filename, text))

        return inputs


    # ======================================================
    # MAIN PIPELINE
    # ======================================================
    logger.info("Loading invoice images from LOCAL + S3...")

    raw_inputs = load_local_inputs() + load_s3_inputs()

    logger.info(f"Found {len(raw_inputs)} invoices.")

    saved = 0

    for filename, invoice_text in raw_inputs:

        prompt = build_prompt(invoice_text)

        logger.info(f"Running Phi-3 → {filename}")

        llm_text = generator(
            prompt,
            max_new_tokens=900,
            do_sample=False,
            temperature=0
        )[0]["generated_text"]

        # OUTPUT FORMAT (only raw_text)
        output_data = {
            "raw_text": llm_text
        }

        save_name = filename.rsplit(".", 1)[0] + ".json"
        save_path = os.path.join(OUTPUT_DIR, save_name)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved → {save_path}")
        saved += 1


    logger.info("===================================")
    logger.info(f"Saved: {saved} processed invoices")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("===================================")

# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    run_parsing_into_structured_json()
# -------------------------------------------------