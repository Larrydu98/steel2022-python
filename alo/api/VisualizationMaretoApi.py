'''
plateYieldStaisticsApi
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
import pandas as pd
# from ..controller.Temperature2DAndFqcController import Temperature2DExtract
# from ..controller.Temperature2DAndFqcController import getTemperature2DQuantileController
from ..controller.VisualizationMaretoController import getMaretoDataController
from ..controller.VisualizationMaretoController import getDataMaretoStationDataController
from ..controller.VisualizationMaretoController import getSixDpictureUpDownQuantileController
from ..controller.VisualizationMaretoController import MaretoLineMergeController

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class SixDpictureUpDownQuantile(Resource):
    '''
    SixDpictureUpDownQuantile
    '''
    def get(self,startTime,endTime,startUpid,endUpid):
        """
        get
        ---
        tags:
          - 可视化马雷图部分
        parameters:
            - in: path
              name: startUpid
              required: true
              description: 开始钢板upid
              type: string
            - in: path
              name: endUpid
              required: true
              description: 结束钢板upid
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}

        MaretoData = getMaretoDataController().run(startTime,endTime)

        UpQuantileData,DownQuantileData = getSixDpictureUpDownQuantileController().run(MaretoData,startUpid,endUpid)


        return {'UpQuantileData': UpQuantileData.to_json(orient='index'),
                'DownQuantileData': DownQuantileData.to_json(orient='index')}, 200, {'Access-Control-Allow-Origin': '*'}



class MaretoLineMergeApi(Resource):
    '''
    MaretoLineMergeApi
    '''
    def get(self,startTime,endTime,lineCrude,lineCrudeIncrease):
        """
        get
        ---
        tags:
          - 可视化马雷图部分
        parameters:
            - in: path
              name: startUpid
              required: true
              description: 开始钢板upid
              type: string
            - in: path
              name: endUpid
              required: true
              description: 结束钢板upid
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}

        MaretoData = getMaretoDataController().run(startTime,endTime)
        StationData = getDataMaretoStationDataController().run(startTime,endTime)
        MaretoData.set_index('upid', inplace=True)

        mareychart_data_upid = MaretoLineMergeController().run(MaretoData,StationData,lineCrude,lineCrudeIncrease)

        mareychart_data_upid_json = mareychart_data_upid.to_json(orient='index')

        return {'mareychart_data_upid_json': mareychart_data_upid_json}, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(SixDpictureUpDownQuantile, '/v1.0/model/SixDpictureUpDownQuantile/<startTime>/<endTime>/<startUpid>/<endUpid>/')
api.add_resource(MaretoLineMergeApi, '/v1.0/model/MaretoLineMergeApi/<startTime>/<endTime>/<lineCrude>/<lineCrudeIncrease>/')