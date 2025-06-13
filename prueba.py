# """Basic connection example.
# """

import redis
import os

redis_client = redis.Redis(
    host='redis-17150.c261.us-east-1-4.ec2.redns.redis-cloud.com',
    port=17150,
    decode_responses=True,
    username="default",
    password=os.getenv("REDI_PASS")  
)


# Patrones que vamos a revisar
patterns = [
    "checkpoint_blob:*",
    "checkpoint:*",
    "checkpoint_write:*",
    "branch:to:*",
]

def is_essential_key(key: str) -> bool:
    # No borres claves esenciales
    return (
        key.endswith(":current")
        or ":default-thread" in key
        or ":active" in key
    )

def clean_redis():
    total_deleted = 0
    for pattern in patterns:
        print(f"🔍 Buscando claves con patrón: {pattern}")
        for key in redis_client.scan_iter(match=pattern):
            if is_essential_key(key):
                print(f"🛑 Conservando clave esencial: {key}")
                continue
            redis_client.delete(key)
            total_deleted += 1
            print(f"🧹 Eliminada: {key}")
    print(f"\n✅ Total de claves eliminadas: {total_deleted}")

if __name__ == "__main__":
    clean_redis()

# # List all keys in the database
# keys = r.keys('*')
# print("Claves en la base de datos:", keys)

# # Optionally, print values for each key
# for key in keys:
#     value = r.get(key)
#     print(f"{key}: {value}")


# from redis.commands.search import Search
# import os
# from dotenv import load_dotenv
# from redis import Redis

# # Load environment variables
# load_dotenv()


# REDIS_URI = os.getenv("REDIS_URI") 

# client = Redis.from_url(REDIS_URI)
# try:
#     info = client.ft("store").info()  # Reemplaza "store" por el nombre del índice
#     print("✅ El índice existe:", info)
# except Exception as e:
#     print("❌ No existe el índice:", e)