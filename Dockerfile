FROM ghcr.io/library/python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn pydantic

COPY . .

ENV PYTHONPATH=/app/src

RUN mkdir -p /app/data /app/artifacts /app/models /app/plots /app/reports /app/logs

EXPOSE 8000

CMD ["uvicorn", "src.ecomopti.phase5.main:app", "--host", "0.0.0.0", "--port", "8000"]