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

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class Platetypes(Resource):
    '''
    SixDpictureUpDownQuantile
    '''
    def get(self,upid):
        """
        get
        ---
        tags:
          - 可视化
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
        # jsondata=getPlatetypeData('select platetype from dcenter.dump_data where upid='+repr(upid))
        jsondata=getSQLData('SELECT  steelspec FROM dcenter.dump_data where upid='+repr(upid))
        return jsondata[0][0],200, {'Access-Control-Allow-Origin': '*'}
api.add_resource(Platetypes, '/v1.0/model/Platetypes/<upid>/')