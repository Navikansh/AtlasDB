# AtlasDB API service
FROM python:3.11-slim

WORKDIR /app

# System deps needed by sentence-transformers/torch at runtime (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY atlasdb/ ./atlasdb/
COPY configs/ ./configs/

ENV PYTHONUNBUFFERED=1
ENV ATLASDB_DATA_DIR=/app/data
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "atlasdb.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
