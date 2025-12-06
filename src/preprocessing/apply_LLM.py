import os
import json
import re
import time
import logging
from logging.handlers import RotatingFileHandler
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def run_applying_LLM():
    # ======================================================
    # LOGGING SETUP
    # ======================================================
    LOG_DIR = "logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("json_extract_preprocessed")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler(
        LOG_DIR + "json_extract_preprocessed.log", maxBytes=5_000_000, backupCount=3
    )

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # ======================================================
    # CONFIG
    # ======================================================
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    INPUT_DIR = "data/processed_pre"
    OUTPUT_DIR = "data/processed"
    DEBUG_DIR = "data/debug_failed"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    logger.info("--------------------------------------------------")
    logger.info("Starting Invoice JSON Extraction Pipeline")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Input folder:  {INPUT_DIR}")
    logger.info(f"Output folder: {OUTPUT_DIR}")
    logger.info("--------------------------------------------------")

    # ======================================================
    # LOAD MODEL
    # ======================================================
    try:
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
            use_safetensors=True,
        )
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False
        )
        logger.info("Model successfully loaded.")

    except Exception as e:
        logger.error(f"ERROR: Failed to load model {MODEL_NAME}")
        logger.error(str(e))
        raise


    # ======================================================
    # SCHEMA
    # ======================================================
    def make_empty_schema():
        return {
            "invoice_number": "",
            "invoice_date": "",
            "due_date": "",
            "vendor": {"name": "", "address": ""},
            "buyer": {"name": "", "address": "", "email": "", "phone": "", "website": ""},
            "line_items": [],
            "totals": {"subtotal": "", "discount": "", "taxes": "", "total_amount": ""},
            "bank_info": {"name": "", "address": "", "city": "", "state": "", "zip": "", "country": ""},
            "notes": "",
            "terms": ""
        }


    # ======================================================
    # MERGE FUNCTION
    # ======================================================
    def merge(schema_item, json_item):
        if isinstance(schema_item, dict):
            result = {}
            for key, val in schema_item.items():
                if key in json_item:
                    result[key] = merge(val, json_item[key])
                else:
                    result[key] = val
            return result
        elif isinstance(schema_item, list):
            return json_item if isinstance(json_item, list) else []
        else:
            return json_item if json_item not in (None, "") else schema_item


    # ======================================================
    # EXTRACTION FUNCTION
    # ======================================================
    def extract_invoice_json(raw_text, filename):
        schema_json = json.dumps(make_empty_schema(), indent=4)

        prompt = f"""
    You MUST return only valid JSON matching exactly this schema:
    {schema_json}

    Rules:
    - Respond with ONLY JSON in correct format. No text before or after.
    - No markdown. No explanations.
    - All fields must appear.
    - If you cannot extract a value, leave it as an empty string.
    - Ensure all line item quantities are correct.

    Invoice Text:
    {raw_text}
    """

        try:
            output = generator(
                prompt,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0
            )[0]["generated_text"]
        except Exception as e:
            logger.error(f"Model generation failed for file: {filename} — {e}")
            return None

        cleaned = output.strip().replace("```json", "").replace("```", "").strip()

        # Extract JSON via regex
        match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", cleaned)
        if not match:
            logger.warning(f"⚠ No valid JSON object found in output for: {filename}")
            return None

        raw_json = match.group(0)

        try:
            parsed = json.loads(raw_json)
            normalized = merge(make_empty_schema(), parsed)
            return normalized

        except Exception as e:
            logger.error(f"JSON parsing error in file: {filename} — {e}")

            # Save raw output for debugging
            debug_path = os.path.join(DEBUG_DIR, filename.replace(".json", "_raw.txt"))
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            logger.info(f"Raw model output saved to: {debug_path}")
            return None


    # ======================================================
    # PROCESS FILES
    # ======================================================
    processed = 0
    failed = 0

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    total_files = len(files)

    logger.info(f"Found {total_files} JSON files to process.")

    start_time = time.time()

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_text = data.get("raw_text", "")

        if not raw_text.strip():
            logger.warning(f"Empty raw_text — skipped: {filename}")
            failed += 1
            continue

        logger.info(f"Extracting JSON from: {filename}")
        t0 = time.time()

        extracted = extract_invoice_json(raw_text, filename)

        if extracted:
            output_path = os.path.join(OUTPUT_DIR, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(extracted, f, indent=4, ensure_ascii=False)

            elapsed = time.time() - t0
            logger.info(f"Saved structured JSON: {filename} ({elapsed:.2f}s)")
            processed += 1
        else:
            logger.error(f"Extraction failed for: {filename}")
            failed += 1

        time.sleep(0.5)


    # ======================================================
    # SUMMARY
    # ======================================================
    total_time = time.time() - start_time
    logger.info("--------------------------------------------------")
    logger.info("Extraction Complete")
    logger.info(f"Processed successfully: {processed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info("Output directory: " + OUTPUT_DIR)
    logger.info("--------------------------------------------------")
    # ======================================================

# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    run_applying_LLM()
# -------------------------------------------------