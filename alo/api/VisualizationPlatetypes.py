'''
Visualization
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
import pandas as pd
import numpy as np
from ..utils import getSQLData
from ..utils import getFlagArr
import json
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
# from ..controller.VisualizationTsneController import getVisualizationTsne

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class VisualizationPlatetypes(Resource):
    '''
    SixDpictureUpDownQuantile
    '''
    def get(self):
        """
        get
        ---
        tags:
          - 可视化
        parameters:
        responses:
            200:
                description: 执行成功
        """
        # jsondata=getPlatetypeData('select distinct platetype from dcenter.dump_data')
        jsondata=getSQLData('SELECT  distinct productcategory FROM dcenter.l2_m_primary_data')

        # dictvalue={}
        # for i in range(len(jsondata)):
        #   print(jsondata[i][0])
        #   temp=getSQLData('SELECT  COUNT(upid) FROM dcenter.l2_m_primary_data where productcategory='+repr(jsondata[i][0]))
        #   print(temp[0][0])
        #   dictvalue[jsondata[i][0]]=temp[0][0]
        # # print(jsondata)
        # # return dictvalue,200, {'Access-Control-Allow-Origin': '*'}
        return jsondata,200, {'Access-Control-Allow-Origin': '*'}
api.add_resource(VisualizationPlatetypes, '/v1.0/model/VisualizationPlatetypes/')