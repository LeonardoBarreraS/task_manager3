# Task Maistro - Railway Optimized Dockerfile (solo Postgres, sin Redis)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Instala solo dependencias necesarias para Python y Postgres
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Por ejemplo, si necesitaras git o curl: \
    # git \
    # curl \
    # Limpia el cache de apt para reducir el tamaño de la imagen
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir redis
RUN pip install --no-cache-dir -r requirements.txt

RUN pip list

COPY main.py .
COPY task_maistro_production.py .
COPY configuration.py .

# Seguridad: usuario no root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["python", "main.py"]
