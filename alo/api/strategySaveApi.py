'''
strategySave
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.strategySaveController import StrategySaveController
import pika, traceback

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

class StrategySave(Resource):
    '''
    StrategySave
    '''
    def post(self):
        """ 
        post
        ---
        tags:
            - 策略generateRun
        parameters:
            - in: body
              name: body
              schema:
              properties:
                  row_id:
                      type: string
                      default: 1
                      description: 策略id
                  task_id:
                      type: string
                      default: ''
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

            strategy_save_instance = StrategySaveController(row_id)
            strategy_save_instance.run()

        except Exception:
            print(traceback.format_exc())
            queue_msg = json.dumps({
                'task_id': task_id,
                'error': '策略不可执行，请检查'
            })

        credentials = pika.PlainCredentials('root', 'woshimima')
        parameters = pika.ConnectionParameters('rabbitmq', 5672, '/', credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        channel.queue_declare(queue='i2studio-endStrategySave', durable=True)

        channel.basic_publish(exchange='',
                            routing_key='i2studio-endStrategySave',
                            body=queue_msg,
                            properties=pika.BasicProperties(
                                delivery_mode=2, # make message persistent
                            ))
        print(" [x]i2studio-endStrategySave:  %r" % row_id)
        connection.close()

        return {'code': 200}

api.add_resource(StrategySave, '/v1.0/strategy/strategySave')
