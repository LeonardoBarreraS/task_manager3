# import redis
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()


# redis_url = os.getenv("REDIS_URI") 

# try:
#     client = redis.Redis.from_url(redis_url)
#     client.ping()
#     print("✅ Conexión exitosa a Redis")
# except Exception as e:
#     print("❌ Error al conectar con Redis:", e)

from redis import Redis
from redis.commands.search.field import TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search import Search
import os

# Conéctate a Redis (ajusta con tu URI y puerto)
client = Redis(
    host="redis-17150.c261.us-east-1-4.ec2.redns.redis-cloud.com",
    port=17150,
    password=os.getenv("REDI_PASS") ,  # pon aquí tu contraseña real
    decode_responses=True
)

# Define el índice (tipo HASH, campo "key" como texto)
try:
    client.ft("store").create_index(
        fields=[TextField("key")],
        definition=IndexDefinition(prefix=["store:"], index_type=IndexType.HASH)
    )
    print("✅ Índice 'store' creado exitosamente.")
except Exception as e:
    if "Index already exists" in str(e):
        print("ℹ️ El índice 'store' ya existe.")
    else:
        print("❌ Error al crear el índice:", e)