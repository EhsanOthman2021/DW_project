# Invoice Lookup API

A FastAPI microservice that:
- Loads processed invoice JSON files from `data/processed/`
- Downloads and reads prediction data from `test_with_predictions.csv` stored in AWS S3
- Matches invoices by file name (supports partial matching)
- Merges:
  - Original JSON invoice data
  - `actual_label` (from CSV)
  - `predicted_label` (from CSV)
- Returns the combined result through an API endpoint

---

## 🚀 Running the Server

Start FastAPI using Uvicorn:

```bash
uvicorn src.api.service:app --host 0.0.0.0 --port 8080 --reload
```

API root:

```
http://localhost:8080/
```

---

## 🔗 API Endpoints

### ✔ Root Check
```
GET /
```

### ✔ Predict / Lookup Invoice
```
GET /predict/{file_name}
```

Example:
```
http://localhost:8080/predict/1.jpg
```

The service will:
- Find a matching file inside `data/processed/`
- Read and return its JSON content
- Add `actual_label` and `predicted_label` from the CSV
- Return the result without modifying the original file

---

## 📂 Test Files

These files exist inside `data/processed/` and can be used for testing:

```
31.json
74.json
77.json
test_10.json
test_13.json
test_4.json
```

### Test URLs

```
http://localhost:8080/predict/31
http://localhost:8080/predict/74
http://localhost:8080/predict/77
http://localhost:8080/predict/test_10
http://localhost:8080/predict/test_13
http://localhost:8080/predict/test_4
```

---

## ❗ Error Responses

### File not found in processed folder
```json
{
  "error": "No processed JSON file found for 'name'"
}
```

### File found in processed folder but NOT in CSV
```json
{
  "error": "File exists in processed folder but not found in CSV",
  "file": "name"
}
```

---

## 📡 AWS S3 Configuration

The service automatically downloads:

```
Bucket: my-fatura2-bucket
Key: invoice_model_reports/test_with_predictions.csv
```

After the first download, the file is reused locally.

---

## 🗂 Project Structure

```
project/
│
├── src/
│   └── api/
│       └── service.py
│
├── data/
│   └── processed/
│       ├── 31.json
│       ├── 74.json
│       ├── 77.json
│       ├── test_10.json
│       ├── test_13.json
│       └── test_4.json
│
├── test_with_predictions.csv
└── README.md
```

---

## 🤝 Contributing

Pull requests are welcome.

## 📄 License

MIT License


docker build -f docker/dockerfile -t fatura-api .
