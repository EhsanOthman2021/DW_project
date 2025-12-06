FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --upgrade pip && pip install .

COPY src ./src
COPY data/processed ./data/processed
COPY test_with_predictions.csv ./test_with_predictions.csv

EXPOSE 8080
CMD ["python", "-m", "src.api.service"]
