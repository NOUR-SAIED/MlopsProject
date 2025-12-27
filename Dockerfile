# Stage 1: Build stage (install dependencies, prepare artifacts)
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --prefix=/install -r requirements.txt

# Copy only artifacts and necessary scripts
COPY artifacts/model_trainer/LightGBM.pkl artifacts/data_transformation/preprocessor.pkl ./
COPY app.py predict.py ./

# Stage 2: Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Copy only necessary files for runtime
COPY --from=builder /app/*.pkl /app/
COPY --from=builder /app/*.py /app/

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["python", "app.py"]
