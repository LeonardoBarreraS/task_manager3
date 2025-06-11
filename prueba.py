"""Basic connection example.
"""

import redis

r = redis.Redis(
    host='redis-17150.c261.us-east-1-4.ec2.redns.redis-cloud.com',
    port=17150,
    decode_responses=True,
    username="default",
    password="JS3btoIvwkXz9sI59cLsFsghjEuyEQsT",
)

success = r.set('foo', 'bar')
# True

result = r.get('foo')
print(result)