# """Basic connection example.
# """

# import redis

# r = redis.Redis(
#     host='redis-17150.c261.us-east-1-4.ec2.redns.redis-cloud.com',
#     port=17150,
#     decode_responses=True,
#     username="default",
#     password="JS3btoIvwkXz9sI59cLsFsghjEuyEQsT",
# )

# # List all keys in the database
# keys = r.keys('*')
# print("Claves en la base de datos:", keys)

# # Optionally, print values for each key
# for key in keys:
#     value = r.get(key)
#     print(f"{key}: {value}")


from redis.commands.search import Search
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


REDIS_URI = os.getenv("REDIS_URI") 

client = Redis.from_url(REDIS_URI)
try:
    info = client.ft("store").info()  # Reemplaza "store" por el nombre del índice
    print("✅ El índice existe:", info)
except Exception as e:
    print("❌ No existe el índice:", e)