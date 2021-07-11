'''
plateYieldStaisticsApi
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.modelTransferCsvController import modelTransferCsvController
from ..controller.getPlateYieldStaisticsAndFlagController import getDataPlateAndFlagCRAndFqc
from ..controller.getPlateYieldStaisticsAndFlagController import getDataPlateYieldAndFlag
from ..controller.Temperature2DAndFqcController import Temperature2DExtract
from ..controller.Temperature2DAndFqcController import getTemperature2DQuantileController
from ..controller.Temperature2DAndFqcController import Temperature2DAndFqcPictureOutputController
from ..controller.Temperature2DAndFqcController import Temperature2DAndFqcSimilarityController
from ..controller.getDataController import getDataCcContentController
from ..controller.getDataController import getDataFqcController

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class Temperature2Ddata(Resource):
    '''
    Temperature2Ddata
    '''
    def get(self,upid):
        """
        get
        ---
        tags:
          - 二维温度场与FQC
        parameters:
            - in: path
              name: plateupid
              required: true
              description: 钢板upid
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}

        CcContentDataClass = getDataCcContentController(upid)
        CcContentData = CcContentDataClass.run()
        Temperature2Ddata_before,Temperature2Ddata_after = Temperature2DExtract().run(CcContentData)

        Temp2DUpQuantileData, Temp2DDownQuantileData = getTemperature2DQuantileController().run(Temperature2Ddata_after.drop(['position'], axis=1))

        return {'Temperature2Ddata_before_position': Temperature2Ddata_before.Position.values.tolist(),
                'Temperature2Ddata_before_data': Temperature2Ddata_before.drop(['Position'],axis=1).values.tolist(),
                'Temperature2Ddata_after_position': Temperature2Ddata_after.position.values.tolist(),
                'Temperature2Ddata_after_data': Temperature2Ddata_after.drop(['position'],axis=1).values.tolist(),
                'Temp2DUpQuantileData': Temp2DUpQuantileData,
                'Temp2DDownQuantileData': Temp2DDownQuantileData}, 200, {'Access-Control-Allow-Origin': '*'}


class FQCdataUpid(Resource):
    '''
    FQCdataUpid
    '''
    def get(self,upid):
        """
        get
        ---
        tags:
          - 二维温度场与FQC
        parameters:
            - in: path
              name: upid
              required: true
              description: 钢板upid
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}

        FqcDataClass = getDataFqcController(upid)
        mother_Fqc_Before_Data,mother_Fqc_After_Data = FqcDataClass.run()

        ####FqcBefore的横坐标不准，从1到尾的序列
        return {'FQCdata_before_positionX': mother_Fqc_Before_Data['Unnamed: 0'].values.tolist(),
                'FQCdata_before_positionY': mother_Fqc_Before_Data.columns.values[1:len(mother_Fqc_Before_Data.columns)].tolist(),
                'FQCdata_before_data': mother_Fqc_Before_Data.drop(['Unnamed: 0'],axis=1).values.tolist(),
                'FQCdata_after_positionX': mother_Fqc_After_Data.x2.values.tolist(),
                'FQCdata_after_positionY': mother_Fqc_After_Data.columns.values[1:len(mother_Fqc_After_Data.columns)].tolist(),
                'FQCdata_after_data': mother_Fqc_After_Data.drop(['x2'],axis=1).values.tolist()}, 200, {'Access-Control-Allow-Origin': '*'}


class Temperature2DAndFQCpictureSimilarity(Resource):
    '''
    Temperature2DAndFQCpictureSimilarity
    '''
    def get(self,upid):
        """
        get
        ---
        tags:
          - 二维温度场与FQC
        parameters:
            - in: path
              name: upid
              required: true
              description: 钢板upid
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}

        CcContentDataClass = getDataCcContentController(upid)
        CcContentData = CcContentDataClass.run()

        FqcDataClass = getDataFqcController(upid)
        mother_Fqc_Before_Data,mother_Fqc_After_Data = FqcDataClass.run()

        if isinstance(CcContentData,str) & isinstance(mother_Fqc_After_Data,str):
            Similarity = '该钢板数据不全，无法计算相似性'
        else:
            Temperature2Ddata_before, Temperature2Ddata_after = Temperature2DExtract().run(CcContentData)
            Temp2DUpQuantileData,Temp2DDownQuantileData = getTemperature2DQuantileController().run(Temperature2Ddata_after.drop(['position'], axis=1))

            Temperature2DAndFqcPictureOutputController().run(Temperature2Ddata_before.drop(['Position'], axis=1),Temp2DDownQuantileData,Temp2DUpQuantileData,'Temperature2D')
            Temperature2DAndFqcPictureOutputController().run(mother_Fqc_Before_Data.drop(['Unnamed: 0'],axis=1),-10,10,'Fqc')

            Similarity = Temperature2DAndFqcSimilarityController().run('Fqc','Temperature2D')


        return {'Similarity': Similarity}, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(Temperature2Ddata, '/v1.0/model/Temperature2Ddata/<upid>/')
api.add_resource(FQCdataUpid, '/v1.0/model/FQCdataUpid/<upid>/')
api.add_resource(Temperature2DAndFQCpictureSimilarity, '/v1.0/model/Temperature2DAndFQCpictureSimilarity/<upid>/')
