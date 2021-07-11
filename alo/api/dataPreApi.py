'''
dataPre
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.dataPreController import DataPreController
import pika, traceback

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class DataPre(Resource):
    '''
    DataPre
    '''
    def post(self):
        """ 
        post
        ---
        tags:
            - 数据预处理
        parameters:
            - in: body
              name: body
              schema:
              properties:
                  row_id:
                  type: string
                  default: 1
                  description: 数据预处理
                  task_id:
                  type: string
                  default: 1
                  description: 任务id
              required: true
        responses:
          200:
            description: 执行成功
        """
        parser.add_argument("row_id", type=str, required=True)
        parser.add_argument("task_id", type=str, required=True)
        args = parser.parse_args(strict=True)
        row_id = args["row_id"]
        task_id = args["task_id"]

        try:
            queue_msg = json.dumps({
                'task_id': task_id
            })

            data_pre_instance = DataPreController(row_id)
            result = data_pre_instance.run()

        except Exception:
            print(traceback.format_exc())
            queue_msg = json.dumps({
                'task_id': task_id,
                'error': '任务执行失败'
            })

        credentials = pika.PlainCredentials('root', 'woshimima')
        parameters = pika.ConnectionParameters('rabbitmq', 5672, '/', credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        channel.queue_declare(queue='i2studio-endDataPre', durable=True)

        channel.basic_publish(
            exchange='',
            routing_key='i2studio-endDataPre',
            body=queue_msg,
            properties=pika.BasicProperties(
                delivery_mode=2, # make message persistent
            )
        )
        print(" [x]endDataPre: %r" % row_id)
        connection.close()

        return {'code': 200}

api.add_resource(DataPre, '/v1.0/model/dataPre')
