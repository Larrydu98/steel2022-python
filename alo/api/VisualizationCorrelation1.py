'''
Visualization
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
import pandas as pd
import numpy as np
from ..utils import getData
from ..utils import getFlagArr
import json
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn import metrics

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class VisualizationCorrelation(Resource):
    '''
    SixDpictureUpDownQuantile
    '''
    def get(self,startTime,endTime):
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
        ismissing={'all_processes_statistics_ismissing':True}
        selection=['all_processes_statistics','all_processes_statistics_ismissing','cool_ismissing','fu_temperature_ismissing','m_ismissing','fqc_ismissing']
        data = getData(['all_processes_statistics','all_processes_statistics_ismissing','cool_ismissing','fu_temperature_ismissing','m_ismissing','fqc_ismissing'], ismissing, [], [], [], [startTime,endTime], [], [], '', '')
        columnslabel=data[0][0]['columns']
        len1=len(data)
        processdata=[]
        fault=[]
        jsondata={'data':[]}
        sortData = {}
        for i in range(len1):
            processdata.append(data[i][0]['data'])
            temp=[]
            for j in range(1,len(selection)):
                temp.append(data[i][j])
            fault.append(temp)
        processdata=np.array(processdata)
        processdata=processdata.swapaxes(0,1)
        fault=np.array(fault)
        fault=fault.swapaxes(0,1)
        nmi_matrix = np.zeros([len(processdata),len(fault)]) #初始化互信息矩阵
        for i in range(len(processdata)):
            for j in range(len(fault)):
                nmi_matrix[i][j] = metrics.normalized_mutual_info_score(processdata[i], fault[j])
        result=np.abs(nmi_matrix.swapaxes(0,1)).tolist()
        faultSum = np.zeros(len(processdata))
        for i in range(len(result)):
            jsondata['data'].append({'fault'+repr(i):result[i]})
            faultSum = np.array(result[i]) + faultSum
            sortData['fault'+repr(i)] = result[i]
            # print(len(result[i]))
        jsondata['label']=columnslabel
        # print(len(columnslabel))

        dataIndex = np.array(range(len(processdata)))

        sortDataFrame = {'faultSum': faultSum, 'dataIndex': dataIndex}
        sortDataFrame = pd.DataFrame(sortDataFrame)
        sortDataFrame = sortDataFrame.sort_values(by="faultSum",ascending=True)
        indexArr = sortDataFrame['dataIndex'].values

        sortData['label'] = columnslabel
        sortData = pd.DataFrame(sortData)

        sortData = sortData.loc[indexArr]
        sortData = sortData.to_dict('list')

        return sortData, 200, {'Access-Control-Allow-Origin': '*'}
api.add_resource(VisualizationCorrelation, '/v1.0/model/VisualizationCorrelation/<startTime>/<endTime>/')