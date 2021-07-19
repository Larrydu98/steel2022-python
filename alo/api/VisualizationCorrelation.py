'''
Visualization
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
import pandas as pd
import numpy as np
from ..utils import getData
from .singelSteel import modeldata_for_corr
from ..utils import getFlagArr
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn import metrics
from .singelSteel import no_category_data_names, no_category_data_names_without_cooling

import warnings
warnings.filterwarnings("ignore")

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


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
        #计算pearson相关系数矩阵
        # def corr2_coeff(A, B):
        #     A_mA = A - A.mean(1)[:, None]
        #     B_mB = B - B.mean(1)[:, None]

        #     ssA = (A_mA**2).sum(1)
        #     ssB = (B_mB**2).sum(1)

        #     return np.dot(A_mA, B_mB.T) / 0.1+np.sqrt(np.dot(ssA[:, None],ssB[None]))

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


        # ismissing={'all_processes_statistics_ismissing':True}
        # selection=['','','','','m_ismissing','fqc_ismissing']
        # data = getData(['all_processes_statistics','all_processes_statistics_ismissing','cool_ismissing','fu_temperature_ismissing','m_ismissing','fqc_ismissing'], ismissing, [], [], [], [startTime,endTime], [], [], '', '')
        selection = ['', '', '', '', 'status_rolling', 'status_fqc']
        len1=len(data)
        processdata=[]
        fault=[]
        for i in range(len1):
            proc_value = []
            if status_cooling == 0 and data[i][4] == 0:
                for key in _data_name:
                    d = data[i][0][key]
                    proc_value.append(d)
                processdata.append(proc_value)

                temp = []
                for j in range(1,len(selection)):
                    temp.append(data[i][j])
                fault.append(temp)
        processdata = pd.DataFrame(processdata).dropna(axis=0, how='any')
        processdata = np.array(processdata)
        processdata=processdata.swapaxes(0,1)
        fault=np.array(fault)
        fault=fault.swapaxes(0, 1)
        # nmi_matrix = np.zeros([len(processdata),len(processdata)]) #初始化互信息矩阵
        # for i in range(len(processdata)):
        #     for j in range(len(processdata)):
        #         nmi_matrix[i][j] = metrics.normalized_mutual_info_score(processdata[i], processdata[j])

        corrdata=np.corrcoef(processdata)

        res = {
            'label': _data_name,
            'corr': corrdata.tolist(),
            # 'nmi': nmi_matrix.tolist()
        }
        return res, 200, {'Access-Control-Allow-Origin': '*'}

api.add_resource(VisualizationCorrelation, '/v1.0/model/VisualizationCorrelation/<startTime>/<endTime>/<limit>')