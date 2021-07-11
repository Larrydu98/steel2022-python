'''
modelPredictCsvAllFlagApi
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.getDataController import getDataUpidClassController
from ..controller.modelPredictCsvAllFlagController import modelPredictCsvAllFlagForOneplateController

import pika, traceback
import pandas as pd
import numpy as np
parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')

class modelPredictCsvAllFlagForOneplate(Resource):
    '''
    modelPredictCsvAllFlagForOneplate
    '''
    def get(self,onePlateUpid):
        """
        get
        ---
        tags:
            - 算法选择
        parameters:
            - in: path
              name: onePlateUpid
              required: true
              description: 一块钢板的Upid
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}
        modelInstance = modelPredictCsvAllFlagForOneplateController(onePlateUpid)
        modelFit = modelInstance.run()

        return {'output': modelFit}, 200, {'Access-Control-Allow-Origin': '*'}


class modelPredictCsvAllFlagForOneplate_test(Resource):
    '''
    modelPredictCsvAllFlagForOneplate_test
    '''
    def get(self,onePlateUpid):
        """
        get
        ---
        tags:
            - 算法选择
        parameters:
            - in: path
              name: onePlateUpid
              required: true
              description: 一块钢板的Upid
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}
        modelFit = {'model_bend_predict':[0.0654,0.9346],
                    'model_thinkness_abnormal_predict':[0.1325,0.8675],
                    'model_horizon_wave_predict':[0.0498,0.9532],
                    'model_left_wave_predict':[0.0015,0.9985],
                    'model_right_wave_predict':[0.0972,0.9028]}

        return {'output': modelFit}, 200, {'Access-Control-Allow-Origin': '*'}

api.add_resource(modelPredictCsvAllFlagForOneplate, '/v1.0/model/modelPredictCsvAllFlagForOneplate/<onePlateUpid>/')
api.add_resource(modelPredictCsvAllFlagForOneplate_test, '/v1.0/model/modelPredictCsvAllFlagForOneplate_test/<onePlateUpid>/')