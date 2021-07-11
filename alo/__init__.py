'''
导入
'''
# import pymysql
from flask import Flask
from flasgger import  Swagger
# from opbeat.contrib.flask import Opbeat
from .api import api_blueprint, fpika
# from .models import db
# from ..config.py import myConfig

# pymysql.install_as_MySQLdb()


app = Flask(__name__)
app.config.from_object('config')
app.register_blueprint(api_blueprint)

# rabbitmq_host = rabbitmq_host
# mysql_host = mysql_host

# db.init_app(app)
# db.app = app

# opbeat = Opbeat(
#     organization_id='d0dd3d8307584b5292a748fe4452acba',
#     app_id='e7938f057a',
#     secret_token='0a0ebf600afbf3493cff87d0ca490aa7a8b2d1f8',
# )
# fpika.init_app(app)

# opbeat.init_app(app)

# fpika = FPika(app)

swagger = Swagger(app)

# if __name__ == '__main__':
#   #app.debug = app.config["DEBUG"]
#   # 本地监听
#   # app.run()
#   # 监听公网ip
#   app.run(host='0.0.0.0')
