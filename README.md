# DW Project – Invoice Data Pipeline, ML Model, and API Service

## 📌 Overview
This project provides an end‑to‑end **data engineering and machine learning architecture** for processing invoice data, training predictive models, and exposing results through a **FastAPI-based service**, fully containerized with Docker and orchestrated with docker-compose.

The system is designed to:
- Process raw invoice data (PDF/PNG)
- Clean and transform the data into structured JSON
- Train an ML model using DVC
- Produce evaluation reports and predictions
- Serve predictions and invoice metadata via a Fast API
- Deploy the API using Docker + docker-compose
- Optionally download prediction files from AWS S3

---

## 🧱 Repository Structure
```
DW_project/
├── config.json
├── dvc.yaml
├── main_pipline.py
│
├── data/
│   ├── processed/
│   ├── raw/
│   └── predictions/
│
├── models/
├── reports/
│
├── src/
│   └── api/
│       └── service.py
│
├── dockerfile
└── docker-compose.yml
```

---

## 🔄 Data & ML Pipeline (DVC)
The ML workflow is orchestrated via **DVC** and includes:

### **1. Preprocessing**
Produces:
```
data/processed/*.json
```

### **2. Training**
Outputs:
```
models/model_xgb.json
models/scaler.pkl
```

### **3. Evaluation**
Outputs:
```
reports/accuracy.html
reports/roc_curve.html
```

### **4. Predictions**
Creates:
```
test_with_predictions.csv
```

---

## 🧠 Machine Learning Model
The trained XGBoost model predicts invoice-level classification labels using engineered features extracted from processed invoice metadata.

---

## 🌐 API Service (FastAPI)
Located in:
```
src/api/service.py
```

### Responsibilities:
- Load processed JSON files
- Load `test_with_predictions.csv`
- Merge metadata + predictions
- Serve REST endpoints:
  - `GET /`
  - `GET /predict/{file_name}`

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
---

## ☁️ AWS S3 Integration
docker-compose automatically downloads:
```
s3://my-fatura2-bucket/invoice_model_reports/test_with_predictions.csv
```

---

## 🐳 Deployment
### Dockerfile
Runs:
```
uvicorn src.api.service:app --host 0.0.0.0 --port 8080
```

### docker-compose
- Injects AWS keys
- Mounts `data/processed/`
- Downloads predictions from S3
- Runs container on **port 8080**

---

## ▶️ How to Run

# Install dependencies using the pyproject.toml file
```
pip install .
```


### 2. Run the ML pipeline
```
dvc repro
```

### 3. Run FastAPI locally
```
uvicorn src.api.service:app --reload
```

### 4. Docker build
```
docker build -t dw_api .
```

### 5. Docker-compose
```
docker-compose up --build
```

---

## 🙌 Author
Developed by **Ehsan Othman** as part of the DW Project.

---
