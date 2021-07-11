'''
createInProcess
'''
# https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
import pandas as pd
import numpy as np
import json
from sklearn import preprocessing
import scipy.io as scio
from scipy.stats import f
from scipy.stats import norm
import matplotlib.mlab as mlab
from ...methods.baogangPlot.PCA import PCATEST
from ...methods.baogangPlot.dataInKind import dataInKind
from ...methods.baogangPlot.setKindMethod import setKindMethod
from ...methods.baogangPlot.prepProData import prepProData

class createInProcess:
    '''
    createInProcess
    '''

    def __init__(self, upid, feature):
        self.upid = upid
        self.feature = feature
        self.paramsIllegal = False
        try:
            # 如果以后加入参数，在这里加入
            print('createInProcess方法暂时没有参数')
        except Exception:
            self.paramsIllegal = True

    def run(self, custom_input):
        '''
        run
        '''
        # feature = custom_input['feature']
        data = custom_input['data']
        targetData = custom_input['targetData']
        # mostId = custom_input['mostId']
        plateKindMess = setKindMethod().general_call({
            'data':targetData,
            'upid': self.upid
        })
        plateKindMess['data'] = targetData
        mostId = dataInKind().general_call(plateKindMess)['finalResult']
        prepedData = prepProData().general_call({'data': data,
                                        'targetData': targetData,
                                        'feature': self.feature,
                                        'kindUpid': mostId,
                                        'upid': self.upid})
        goodData = prepedData['goodData']
        badData = prepedData['badData']
        testBoardData = prepedData['testBoardData']

        result = []
        count = 0
        n1 = len(goodData)

        y_np = np.zeros([100,1001])
        data_np = np.zeros([100,1001])
        boardDatagood = np.array([])
        for item in mostId:
            if str(item) in list(goodData.keys()):
                boardData =  np.array(goodData[str(item)]).flatten()
                boardDataWash = [i for i in boardData]
                boardDatagood = boardDataWash
                counts, bin_edges = np.histogram(boardDataWash,bins=1000)
        #         if counts[0] < 40 and counts[2]> 150 and counts[4] < 50 and counts[2]<220 and counts[3]>150 and counts[1]>50:
                (mu, sigma) = norm.fit(boardDataWash)
                y = norm.pdf(bin_edges, mu, sigma)
                data_np[count] = bin_edges
                y_np[count] = y
                count = count + 1

        good_x_temp = bin_edges
        data_df = pd.DataFrame(data_np,columns = y)
        y_df = pd.DataFrame(y_np,columns = bin_edges)

        data_df = data_df.ix[~(data_df==0).all(axis=1), :]
        y_df = y_df.ix[~(y_df==0).all(axis=1), :]

        y_max_np = y_df.quantile(0.92,axis=0).values
        y_min_np = y_df.quantile(0.18,axis=0).values
        boardDatabad = np.array([])
        boardData =  np.array(testBoardData).flatten()
        boardDataWash = [i for i in boardData]
        boardDatabad = boardDataWash
        counts, bin_edges = np.histogram(boardDataWash,bins=1000)
        (mu, sigma) = norm.fit(boardDataWash)
#             plt.hist(boardDataWash,bins=5, normed=1)
#             plt.plot(bin_edges[1:], counts, label=str(item))
        y = norm.pdf(bin_edges, mu, sigma) 
        bad_y = y
        bad_x = bin_edges
        # print(bin_edges)
        # print(y)
        bias_minmax = 0
        bias_average = 0
         
        for i in range(len(bad_y.tolist())):
            very_poor = y_max_np.tolist()[i] - y_min_np.tolist()[i]
            if bad_y.tolist()[i] < y_min_np.tolist()[i]:
                bias_minmax = bias_minmax + abs(bad_y.tolist()[i] - y_min_np.tolist()[i])/very_poor
            if bad_y.tolist()[i] > y_max_np.tolist()[i]:
                biabias_minmaxs = bias_minmax + abs(bad_y.tolist()[i] - y_max_np.tolist()[i])/very_poor
            average = 1/2*very_poor
            bias_average = bias_average + abs(average-bad_y.tolist()[i])/very_poor

            result.append({
                'value': bad_y.tolist()[i],
                'name': bad_x.tolist()[i],
                'l': y_min_np.tolist()[i],
                'u': y_max_np.tolist()[i]
            })
        return result, bias_minmax, bias_average
