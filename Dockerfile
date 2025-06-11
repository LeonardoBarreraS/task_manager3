# Task Maistro - Optimized Dockerfile for Redis Persistence
# Usa una imagen base de Python ligera
FROM python:3.11-slim

# Configura variables de entorno para Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala paquetes de sistema que sean necesarios.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Limpia el cache de apt para reducir el tamaño de la imagen
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copia el archivo de requisitos
COPY requirements.txt .

# --- SOLUCIÓN DEFINITIVA: Instala dependencias en un orden específico y robusto ---

# 1. Asegúrate de que pip esté actualizado.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 2. Instala la librería 'redis' explícitamente y de forma aislada.
# Esto garantiza que el módulo 'redis' esté disponible para LangGraph.
RUN pip install --no-cache-dir redis

# 3. Luego, instala LangGraph (sin el extra [redis] en requirements.txt)
# y todas las demás dependencias desde requirements.txt.
# Esto permite que LangGraph encuentre el módulo 'redis' ya instalado.
RUN pip install --no-cache-dir -r requirements.txt

# --- PASOS DE DEPURACIÓN CRÍTICA: Verifica la instalación y la importación ---
# Estos comandos se ejecutarán durante la construcción de la imagen.
# Revisa los logs de la construcción de Docker muy cuidadosamente.

# Lista todos los paquetes instalados y sus versiones
RUN echo "--- Verificando paquetes instalados ---"
RUN pip list

# Intenta importar los módulos de Redis y LangGraph directamente en un script Python.
# Si esto falla aquí, el problema es en la instalación/disponibilidad durante la construcción.
RUN echo "--- Intentando importar redis y langgraph.checkpoint.redis ---"
RUN python -c "import redis; print(f'Redis installed: {redis.__version__}'); from langgraph.checkpoint import redis as lr_redis; print('langgraph.checkpoint.redis import successful!')"
RUN echo "--- Verificación de importación completa ---"

# --- DEPURACIÓN AVANZADA: Inspeccionar la ruta de instalación ---
# Lista el contenido de la carpeta de LangGraph para ver si 'redis.py' o 'redis/' está presente.
RUN echo "--- Contenido de langgraph.checkpoint/ ---"
RUN ls -la /usr/local/lib/python3.11/site-packages/langgraph/checkpoint/
RUN echo "--- Fin de listado ---"


# Copia los archivos de tu aplicación al contenedor
COPY main.py .
COPY task_maistro_production.py .
COPY configuration.py .

# Seguridad: crea un usuario no-root para ejecutar la aplicación
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expone el puerto en el que la aplicación Gradio escuchará
EXPOSE 8080

# Comando para iniciar la aplicación Gradio
CMD ["python", "main.py"]