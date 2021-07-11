'''
deepCNNSinTestController
'''
import numpy as np
from flask import json
from  .getDataController import getDataController
from ..methods.algorithmChooseMethods.DeepCNN import DeepCNN


class deepCNNSinTestController:
    '''
    deepCNNSinTestController
    '''
    def __init__(self):
        self.idIllegal = False

    # 创建模型实例   
    def createModel(self):
        '''
        createModel
        '''
        model = DeepCNN()
        return model
          
    def run(self, inputData):
        '''
        run
        '''
        x_test_wide_best = np.array(inputData['x_test_wide'])
        x_test_deep1_best = np.array(inputData['x_test_deep1'])
        x_test_deep2_best = np.array(inputData['x_test_deep2'])
        x_test_deep3_best = np.array(inputData['x_test_deep3'])
        x_test_deep4_best = np.array(inputData['x_test_deep4'])
        x_test_deep5_best = np.array(inputData['x_test_deep5'])
        x_test_deep6_best = np.array(inputData['x_test_deep6'])
        y_test_best = np.array(inputData['y_test'])
        y_test_raw = np.array(inputData['y_test_raw'])
        model_dict_history = model_dict_history   ####改一下
        upid_train = inputData['upid_train']
        upid_test = inputData['upid_test']
        upin_all_after_filter = pd.concat(upid_train, upid_test)
        upid_all_before_filter = getDataController(6, '', '')
        model_test = self.createModel()
        # 对一块钢板数据进行预测
        for upid in inputData.upid_test:

            upid_test_one = upid

            upid_test_drop = upin_all_after_filter.reset_index()
            upid_test_drop = upid_test_drop.drop(['index'], axis=1)

            upid_all_before_filter_drop = upid_all_before_filter.reset_index()
            upid_all_before_filter_drop = upid_all_before_filter_drop(['index'], axis=1)

            if upid_test_drop[upid_test_drop['upid'] == upid_test_one].empty:
                result[upid] = '未经预处理的钢板数据号'

            else:
                test_index = upid_test_drop[upid_test_drop['upid'] == upid_test_one].index.values[0]
                x_test_wide_one = x_test_wide_best[test_index].reshape(1, x_test_wide_best.shape[1],x_test_wide_best.shape[2],x_test_wide_best.shape[3])
                x_test_deep1_one = x_test_deep1_best[test_index].reshape(1, x_test_deep1_best.shape[1],x_test_deep1_best.shape[2],x_test_deep1_best.shape[3])
                x_test_deep2_one = x_test_deep2_best[test_index].reshape(1, x_test_deep2_best.shape[1],x_test_deep2_best.shape[2],x_test_deep2_best.shape[3])
                x_test_deep3_one = x_test_deep3_best[test_index].reshape(1, x_test_deep3_best.shape[1],x_test_deep3_best.shape[2],x_test_deep3_best.shape[3])
                x_test_deep4_one = x_test_deep4_best[test_index].reshape(1, x_test_deep4_best.shape[1],x_test_deep4_best.shape[2],x_test_deep4_best.shape[3])
                x_test_deep5_one = x_test_deep5_best[test_index].reshape(1, x_test_deep5_best.shape[1],x_test_deep5_best.shape[2],x_test_deep5_best.shape[3])
                x_test_deep6_one = x_test_deep6_best[test_index].reshape(1, x_test_deep6_best.shape[1],x_test_deep6_best.shape[2],x_test_deep6_best.shape[3])
                y_test_one = y_test_best[test_index]

                model_test_predict_one = model_test.predict([x_test_wide_one, x_test_deep1_one, x_test_deep2_one, x_test_deep3_one,x_test_deep4_one, x_test_deep5_one, x_test_deep6_one])
            if upid_all_before_filter_drop[upid_all_before_filter_drop['upid'] == upid_test_one].empty:
                result[upid] = '当前数据不存在的钢板数据号'

            else:
                test_index = upid_all_before_filter_drop[upid_all_before_filter_drop['upid'] == upid_test_one].index.values[0]
                result[upid] = upid_all_before_filter_drop[test_index]
  
        return result
