'''
strategyRun
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.strategyRunController import StrategyRunController
import pika, traceback
import copy

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

class StrategyRun(Resource):
    '''
    StrategyRun
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
                  input_obj:
                      type: string
                      default: ''
                      description: 输入字符串形式
              required: true
        responses:
            200:
                description: 执行成功
        """
        parser.add_argument("row_id", type=str, required=True)
        parser.add_argument("task_id", type=str, required=True)
        parser.add_argument("input_obj", type=str, required=True)
        args = parser.parse_args(strict=True)
        row_id = args["row_id"]
        task_id = args["task_id"]
        input_obj = json.loads(args["input_obj"])

        try:
            strategy_run_instance = StrategyRunController(row_id)
            strategyOutput = copy.copy(strategy_run_instance.run(input_obj))
            for k,v in strategyOutput.items():
                if hasattr(v, 'pridict'):
                    strategyOutput[k] = '策略输出模型成功'
                else:
                    strategyOutput[k] = v.tolist()
            print(strategy_run_instance.run(input_obj))
            print(type(strategy_run_instance.run(input_obj)))
            queue_msg = json.dumps({
                'task_id': task_id,
                'outputResult': strategyOutput
            })

        except Exception:
            print(traceback.format_exc())
            queue_msg = json.dumps({
                'task_id': task_id,
                'error': '策略执行出现问题'
            })

        credentials = pika.PlainCredentials('root', 'woshimima')
        parameters = pika.ConnectionParameters('rabbitmq', 5672, '/', credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        channel.queue_declare(queue='i2studio-endStrategyRun', durable=True)

        channel.basic_publish(exchange='',
                            routing_key='i2studio-endStrategyRun',
                            body=queue_msg,
                            properties=pika.BasicProperties(
                                delivery_mode=2, # make message persistent
                            ))
        print(" [x]i2studio-endStrategyRun:  %r" % row_id)
        connection.close()

        return {'code': 200}

api.add_resource(StrategyRun, '/v1.0/strategy/strategyRun')
