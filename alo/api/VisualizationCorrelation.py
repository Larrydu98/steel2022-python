'''
Visualization
'''
from flask_restful import Resource, reqparse
from . import api
import pandas as pd
import numpy as np
from .singelSteel import modeldata_for_corr
from sklearn import metrics
from .singelSteel import no_category_data_names, no_category_data_names_without_cooling

import warnings
warnings.filterwarnings("ignore")

parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class VisualizationCorrelation(Resource):
    '''
    SixDpictureUpDownQuantile
    '''
    def post(self, startTime, endTime, limit):
        """
        get
        ---
        tags:
          - 可视化相关性分析
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

        data, status_cooling = modeldata_for_corr(parser,
                                         ['dd.stats', 'dd.status_stats', 'dd.status_furnace', 'dd.status_rolling', 'dd.status_cooling', 'dd.status_fqc'],
                                         startTime,
                                         endTime,
                                         limit)
        _data_name = []
        if status_cooling == 0:
            _data_name = no_category_data_names
        else:
            _data_name = no_category_data_names_without_cooling

        processdata=[]
        for i in range(len(data)):
            proc_value = []
            if status_cooling == 0 and data[i][4] == 0:
                for key in _data_name:
                    d = data[i][0][key]
                    proc_value.append(d)
                processdata.append(proc_value)

        processdata = pd.DataFrame(processdata).dropna(axis=0, how='any')
        processdata = np.array(processdata)
        processdata=processdata.swapaxes(0,1)

        # nmi_matrix = np.zeros([len(processdata),len(processdata)]) #初始化互信息矩阵
        # for i in range(len(processdata)):
        #     for j in range(len(processdata)):
        #         if j < i:
        #             nmi_matrix[i][j] = metrics.normalized_mutual_info_score(processdata[i], processdata[j])

        corrdata=np.corrcoef(processdata)

        res = {
            'label': _data_name,
            'corr': corrdata.tolist(),
            # 'nmi': nmi_matrix.tolist()
        }
        return res, 200, {'Access-Control-Allow-Origin': '*'}

api.add_resource(VisualizationCorrelation, '/v1.0/model/VisualizationCorrelation/<startTime>/<endTime>/<limit>')