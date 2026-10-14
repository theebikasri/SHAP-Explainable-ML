FROM python:3.12-slim

WORKDIR /app

# Install build tools needed by some Python packages (e.g. shap, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and all required artifacts
COPY app.py api.py train.py ./
COPY src/ ./src/
COPY models/ ./models/
COPY dataset/ ./dataset/
COPY outputs/ ./outputs/

# Ports used by the two services
EXPOSE 8501 8000
