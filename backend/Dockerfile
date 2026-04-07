FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user and logs directory
RUN adduser --disabled-password --gecos '' --uid 1000 appuser
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

COPY . .

# Run as non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
