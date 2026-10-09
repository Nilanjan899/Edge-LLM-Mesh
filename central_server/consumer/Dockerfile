FROM python:3.11-slim

WORKDIR /app

# Install consumer dependencies (subset of api requirements)
RUN pip install --no-cache-dir pika prometheus-client

COPY consumer/process_metrics.py .

CMD ["python", "process_metrics.py"]
