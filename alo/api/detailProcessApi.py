'''
detailProcessApi
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.baogangPlot.createInProcess import createInProcess
import pika, traceback
import pandas as pd
import numpy as np
import os
path = os.getcwd()
# path1=os.path.abspath('.') + 'wide_117_data_flag_20190831.csv'
# path2=os.path.abspath('.') + 'steelspec_data20180902.csv'
# filePath_1 = r'D:\document\BAOGANGDATA\model_raw_data_190523.json'
parser = reqparse.RequestParser(trim=True, bundle_errors=True)
def getData():
    filePath_1 = "/usr/src/data/mother_fqc_before/10.json"
    # filePath_1 = r'/alo/data/model_raw_data_190523.json'
    # filePath_2 = r'D:\document\BAOGANGDATA\plate_type_distribution190530.csv'
    filePath_2 = "/usr/src/data/mother_fqc_before/11.csv"
    # filePath_2 = r'/alo/data/plate_type_distribution190530.csv'
    # filePath3 = r'D:\document\ALO_PYTHON\alo\api\wide_117_data_flag_20190831.csv'
    # filePath4 = r'D:\document\ALO_PYTHON\alo\api\wide_117_data_flag_20190831.csv'

   
    # targetData_dir = pd.read_csv(filePath_2)
    targetData_dir = pd.read_csv("/usr/src/data/mother_fqc_before/11.csv")
    targetData_dir = targetData_dir.drop_duplicates(['upid'], keep='last').reset_index().drop(['index'], axis=1)
    targetData_dir[['upid']] = targetData_dir[['upid']].astype('str')
    targetData_dir = targetData_dir.set_index('upid')
    targetData = targetData_dir.dropna()
    # data = pd.read_json(filePath_1)
    data = pd.read_json("/usr/src/data/mother_fqc_before/10.json")
    data = data.drop_duplicates(['upid'], keep='last').reset_index().drop(['index'], axis=1)
    data[['upid']] = data[['upid']].astype('str')
    data[['upid']] = data[['upid']].astype('str')
    data = data.set_index('upid')
    result = {
      'data': data,
      'targetData': targetData
    }
    return  result

# # 根目录
# @app.route('/')


class detailProcessApi(Resource):
    '''
    detailProcessApi
    '''
    def post(self):
        """ 
        post
        ---
        tags:
            - 诊断视图工序数据
        parameters:
            - in: body
              name: body
              schema:
              properties:
                  upid:
                #   type: string
                #   default: 1
                #   description: 数据预处理
                  task_id:
                #   type: string
                #   default: 1
                #   description: 任务id
              required: true
        responses:
          200:
            description: 执行成功
        """
        parser.add_argument("upid", type=str, required=True)
        parser.add_argument("feature", type=str, required=True)
        args = parser.parse_args(strict=True)
        upid = args["upid"]
        feature = args['feature']
        print(feature)
        # task_id = args["task_id"]
        print('*************')
        print(upid)
        result =  []
        myresult = getData()
        data = myresult['data']
        targetData = myresult['targetData']
        try:
            # queue_msg = json.dumps({
            #     'task_id': task_id
            # })

            createInProcess_instance = createInProcess(upid, feature)
            result, bias_minmax, bias_average = createInProcess_instance.run({
                'targetData': targetData,
                'data': data,
                'feature': feature
            })

        except Exception:
            print(traceback.format_exc())
            # queue_msg = json.dumps({
            #     'task_id': task_id,
            #     'error': '任务执行失败'
            # })

        # credentials = pika.PlainCredentials('root', 'woshimima')
        # parameters = pika.ConnectionParameters(rabbitmq_host, 5672, '/', credentials)
        # connection = pika.BlockingConnection(parameters)
        # channel = connection.channel()

        # channel.queue_declare(queue='i2studio-endDataPre', durable=True)

        # channel.basic_publish(
        #     exchange='',
        #     routing_key='i2studio-endDataPre',
        #     body=queue_msg,
        #     properties=pika.BasicProperties(
        #         delivery_mode=2, # make message persistent
        #     )
        # )
        # print(" [x]endDataPre: %r" % row_id)
        # connection.close()

        # return {'code': 200}
        # print(result)
        return {'result': result, 'bias_minmax': bias_minmax, 'bias_average': bias_average}, 200, {'Access-Control-Allow-Origin': '*'}

api.add_resource(detailProcessApi, '/v1.0/baogangPlot/diagnosesdetaildata')
