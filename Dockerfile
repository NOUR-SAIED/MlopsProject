# Dockerfile - Use Python 3.11 to match Jenkins
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements only
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Copy ALL source code BEFORE installing requirements
COPY . .

# NOW install requirements (including local package)
RUN pip install -r requirements.txt

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["python", "app.py"]