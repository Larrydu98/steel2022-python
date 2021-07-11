'''
modelTransferController
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.modelTransferController import modelTransferController
import pika, traceback
import pandas as pd
import numpy as np
parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class modelTransferApi(Resource):
    '''
    modelTransferController
    '''
    def get(self,startTime,endTime):
        """ 
        get
        ---
        tags:
            - 算法选择
        parameters:
            - in: path
              name: startTime
              required: true
              description: 开始时间
              type: string
            - in: path
              name: endTime
              required: true
              description: 结束时间
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}
        modelInstance = modelTransferController(startTime,endTime)
        modelFit = modelInstance.run()

        return {'username': modelFit}, 200, {'Access-Control-Allow-Origin': '*'}

api.add_resource(modelTransferApi, '/v1.0/model/modelTransferController/<startTime>/<endTime>/')
