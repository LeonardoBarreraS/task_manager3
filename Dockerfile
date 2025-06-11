# Task Maistro - Simplified Dockerfile for Redis + Railway + Gradio
FROM python:3.11-slim

# Variables de entorno para un comportamiento predecible de Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo
WORKDIR /app

# Instala solo lo esencial del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia y actualiza pip + instala dependencias
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copia los archivos de la aplicación
COPY main.py .
COPY task_maistro_production.py .
COPY configuration.py .

# Crea un usuario no-root por seguridad
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expone el puerto que Gradio usará
EXPOSE 8080

# Comando para ejecutar tu app
CMD ["python", "main.py"]