'''
配置文件
'''

# from alo import rabbitmq_host, mysql_host

# 启动Flask调试模式
DEBUG = True

# config.py
FLASK_PIKA_PARAMS = {
    'host': 'localhost',
    'username': 'guest',  # convenience param for username
    'password': 'guest',  # convenience param for password
    'port': 5672  # amqp server port
    # 'virtual_host':'/'   #amqp vhost
}

# optional pooling params
FLASK_PIKA_POOL_PARAMS = {
    'pool_size': 8,
    'pool_recycle': 30
}

SWAGGER = {
    'title': '氧化铝算法API文档',
    'uiversion': 2
}

# SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:woshimima@' + mysql_host + ':3306/i2studio?charset=utf8mb4'
# SQLALCHEMY_TRACK_MODIFICATIONS = True


# del os
