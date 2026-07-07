FROM python:3.10-slim

# Install light system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (Hugging Face defaults to 7860)
EXPOSE 7860

# Run app
CMD ["python", "app.py"]
