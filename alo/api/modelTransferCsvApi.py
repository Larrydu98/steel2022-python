'''
modelTransferCsvApi
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.getDataController import getDataUpidClassController
from ..controller.modelTransferCsvController import modelTransferCsvController
from ..controller.modelTransferCsvController import modelTransferCsvforSomePlate
from ..controller.modelTransferCsvController import modelTransferCsvforOnePlate
from ..controller.modelTransferCsvController import modelTransferCsvforPlateByTime
from ..controller.dataFilterCsvAfter import getDataFilterCsvAfterTimeByUpid
from ..controller.deepCNNTestController import getModelWeightsController
from ..controller.getDataController import getDataMOutputPgController
from ..controller.getDataController import getDataMCcTempController
from ..controller.getDataController import getDataP12456TempController
from ..controller.getDataController import getDataPlateThinessPrimaryController
import pika, traceback
import pandas as pd
import numpy as np
parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class modelTransferCsv(Resource):
    '''
    modelTransferCsv
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
        modelInstance = modelTransferCsvController(startTime,endTime)
        modelFit = modelInstance.run()

        return {'username': modelFit}, 200, {'Access-Control-Allow-Origin': '*'}


class modelTransferCsvForSomePlate(Resource):
    '''
    modelTransferCsvForSomePlate
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
        modelInstance = modelTransferCsvforSomePlate(startTime,endTime)
        modelFit = modelInstance.run()

        return {'modelPredict': modelFit}, 200, {'Access-Control-Allow-Origin': '*'}


class modelTransferCsvForOnePlate(Resource):
    '''
    modelTransferCsvForOnePlate
    '''
    def get(self,upid):
        """
        get
        ---
        tags:
            - 算法选择
        parameters:
            - in: path
              name: upid
              required: true
              description: 钢板的upid
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}
        modelInstance = modelTransferCsvforOnePlate(upid)
        modelFit = modelInstance.run()

        return {'modelPredict': modelFit}, 200, {'Access-Control-Allow-Origin': '*'}


class BasicInformationCsvForOnePlate(Resource):
    '''
    BasicInformationCsvForOnePlate
    '''
    def get(self,upid):
        """
        get
        ---
        tags:
            - 算法选择
        parameters:
            - in: path
              name: upid
              required: true
              description: 钢板的upid
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}
        DataBasicInformation = getDataUpidClassController(upid).run()

        return {'DataBasicInformation': DataBasicInformation}, 200, {'Access-Control-Allow-Origin': '*'}

class modelTransferCsvForPlateByTime(Resource):
    '''
    modelTransferCsvForPlateByTime
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
        modelInstance = modelTransferCsvforPlateByTime(startTime,endTime)
        modelFit = modelInstance.run()

        return {'modelPredict': modelFit}, 200, {'Access-Control-Allow-Origin': '*'}


class modelTransferCsvForPlateAfterTimeByUpid(Resource):
    '''
    modelTransferCsvForPlateAfterTimeByUpid
    '''
    def get(self,upid,upidBeforeTimeDay):
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
        upidBeforeTimeDay = int(upidBeforeTimeDay)
        startTime,endTime = getDataFilterCsvAfterTimeByUpid().run(upid,upidBeforeTimeDay)
        modelInstance = modelTransferCsvforPlateByTime(startTime,endTime)
        modelFit = modelInstance.run()

        return {'modelPredict': modelFit}, 200, {'Access-Control-Allow-Origin': '*'}


class getDataMOutputPgApi(Resource):
    '''
    getDataMOutputPgApi
    '''
    def get(self,upid):
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
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}
        PlateThinessPrimary = getDataPlateThinessPrimaryController(upid).run()
        PgData = getDataMOutputPgController(upid).run()

        return {'PgData': PgData,
                'PlateThinessPrimary': PlateThinessPrimary}, 200, {'Access-Control-Allow-Origin': '*'}


class getDataP12456Api(Resource):
    '''
    getDataP12456Api
    '''
    def get(self,upid):
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
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}

        P12456Data = getDataP12456TempController(upid).run()

        return {'PostionX': P12456Data.postionX.values.tolist(),
                'TempP1': P12456Data.TempP1.values.tolist(),
                'TempP2L': P12456Data.TempP2L.values.tolist(),
                'TempP4': P12456Data.TempP4.values.tolist(),
                'TempP5': P12456Data.TempP5.values.tolist(),
                'TempP6': P12456Data.TempP6.values.tolist()}, 200, {'Access-Control-Allow-Origin': '*'}


class getDataMCcTempApi(Resource):
    '''
    getDataMCcTempApi
    '''
    def get(self,upid):
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
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}

        MCcTempData = getDataMCcTempController(upid).run()

        return {'TimeX': MCcTempData.Time.values.tolist(),'TempY': MCcTempData.Temp.values.tolist()}, 200, {'Access-Control-Allow-Origin': '*'}


class getModelWeightsCsvApi(Resource):
    '''
    getModelWeightsCsvApi
    '''
    def get(self):
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
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}

        Model_Weights_list_ave = getModelWeightsController().run()

        return {'Model_Weights_list_ave': Model_Weights_list_ave}, 200, {'Access-Control-Allow-Origin': '*'}


class modelTransferCsvTrainDataClass(Resource):
    '''
    modelTransferCsvTrainDataClass
    '''
    def get(self):
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
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}

        Model_Weights_list_ave = getModelWeightsController().run()
        DataClass = ['单维序列数据','炉温数据','轧制结束后凸度仪测量数据','轧制过程轧制力，弯辊力等控制变量数据','轧制过程平直度，厚度输出数据','冷却出口处二维温度场数据','冷却过程单点测温仪数据']
        DataDimension = ['117X1','6X40','3X100','12X7','10X7','9X100','5X100']

        return {'Model_Weights_list_ave': Model_Weights_list_ave,
                'DataClass': DataClass,
                'DataDimension': DataDimension}, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(modelTransferCsv, '/v1.0/model/modelTransferCsv/<startTime>/<endTime>/')
api.add_resource(modelTransferCsvForSomePlate, '/v1.0/model/modelTransferCsvForSomePlate/<startTime>/<endTime>/')
api.add_resource(modelTransferCsvForOnePlate, '/v1.0/model/modelTransferCsvForOnePlate/<upid>/')
api.add_resource(BasicInformationCsvForOnePlate, '/v1.0/model/BasicInformationCsvForOnePlate/<upid>/')
api.add_resource(modelTransferCsvForPlateByTime, '/v1.0/model/modelTransferCsvForPlateByTime/<startTime>/<endTime>/')
api.add_resource(modelTransferCsvForPlateAfterTimeByUpid, '/v1.0/model/modelTransferCsvForPlateAfterTimeByUpid/<upid>/<upidBeforeTimeDay>/')

api.add_resource(getDataMOutputPgApi, '/v1.0/model/getDataMOutputPgApi/<upid>/')
api.add_resource(getDataMCcTempApi, '/v1.0/model/getDataMCcTempApi/<upid>/')
api.add_resource(getDataP12456Api, '/v1.0/model/getDataP12456Api/<upid>/')
api.add_resource(getModelWeightsCsvApi, '/v1.0/model/getModelWeightsCsvApi/')
api.add_resource(modelTransferCsvTrainDataClass, '/v1.0/model/modelTransferCsvTrainDataClass/')