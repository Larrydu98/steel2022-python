'''
VisualizationMDSApi
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
import numpy as np
import pandas as pd
from ..utils import getData,new_getData
from ..utils import getFlagArr
from ..utils import ref

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class getFlag(Resource):
    '''
    getFlag
    '''
    def get(self,startTime,endTime):
        """
        get
        ---
        tags:
          - 获取钢板标签
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
        tocSelect = [startTime, endTime]
        data, col_names = new_getData(['upid', 'fqc_label','status_fqc'], {'status_stats':True}, [], [], [], tocSelect, [], [], '', '')

        result = {}
        # print(data)
        for item in data:
            label = 0
            if item[2] == 0:
                flags = item[1]['method1']['data']
                if np.array(flags).sum() == 5:
                    label = 1
            elif item[2] == 1:
                label = 404
            result[item[0]] = label

        # print(json)

        return result, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(getFlag, '/v1.0/getFlag/<startTime>/<endTime>/')