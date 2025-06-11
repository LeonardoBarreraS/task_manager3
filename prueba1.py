import redis
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


redis_url = os.getenv("REDIS_URI") 

try:
    client = redis.Redis.from_url(redis_url)
    client.ping()
    print("✅ Conexión exitosa a Redis")
except Exception as e:
    print("❌ Error al conectar con Redis:", e)