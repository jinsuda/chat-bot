import redis


def create_redis_client():
    return redis.Redis(
        host="redis-13883.c54.ap-northeast-1-2.ec2.cloud.redislabs.com",
        port=13883,
        password="FMUVpL5Oy70Yhp1GLqshlLCnHTkT116Y",
    )
