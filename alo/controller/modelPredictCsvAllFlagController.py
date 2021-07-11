'''
modelPredictCsvAllFlagController
'''
import pandas as pd
from  .dataTransform_Tomodel_Fromcsv import dataTransform_OneAbnormal_Fromcsv

from  .deepModelPredictController import deepModelPredict



# class modelPredictCsvAllFlagForOneplateController:
#     '''
#     modelPredictCsvAllFlagForOneplate
#     '''
#     def __init__(self, onePlateUpid):
#         self.onePlateUpid = onePlateUpid
#
#     def run(self):
#         '''
#         run
#         '''
#         afterInstanceFilter = dataFilterCsvAfter()
#         afterMyData = afterInstanceFilter.run()
#         modelInstance = deepCNNTestController()
#         modelData = modelInstance.run(afterMyData)
#         return modelData
# # instance = modelTransferController('2018-09-01 00:00:00', '2018-09-02 00:00:00')
# # instance.run()


class modelPredictCsvAllFlagForOneplateController:
    '''
    modelPredictCsvAllFlagForOneplate
    '''
    def __init__(self, onePlateUpid):
        self.onePlateUpid = onePlateUpid

    def run(self):
        '''
        run
        '''

        data_train_start = '18901000000'
        data_train_end = '19315000000'
        data_test_start = '19315000000'
        data_test_end = '19401000000'

        print('开始提取数据')
        DataInstance = dataTransform_OneAbnormal_Fromcsv(data_train_start ,data_train_end, data_test_start, data_test_end)
        train_data_bend_flag, test_data_bend_flag = DataInstance.run('bend_flag')
        train_data_thinkness_abnormal_flag, test_data_thinkness_abnormal_flag = DataInstance.run('thinkness_abnormal_flag')
        train_data_horizon_wave_flag, test_data_horizon_wave_flag = DataInstance.run('horizon_wave_flag')
        train_data_left_wave_flag, test_data_left_wave_flag = DataInstance.run('left_wave_flag')
        train_data_right_wave_flag, test_data_right_wave_flag = DataInstance.run('right_wave_flag')
        print('提取数据完成')

        print('开始加载模型结构')
        modelInstance = deepModelPredict()
        model_Structure = modelInstance.createModelStructure()
        print('加载模型结构完成')

        print('开始加载模型权重')
        model_bend = modelInstance.createModelWeights(model_Structure,'bend_flag')
        print('1')
        model_thinkness_abnormal = modelInstance.createModelWeights(model_Structure, 'thinkness_abnormal_flag')
        print('2')
        model_horizon_wave = modelInstance.createModelWeights(model_Structure, 'horizon_wave_flag')
        print('3')
        model_left_wave = modelInstance.createModelWeights(model_Structure, 'left_wave_flag')
        print('4')
        model_right_wave = modelInstance.createModelWeights(model_Structure, 'right_wave_flag')
        print('加载模型权重完成')

        print('开始预测')
        model_bend_predict = modelInstance.ModelPredictForOneplate(model_bend, test_data_bend_flag,self.onePlateUpid)
        model_thinkness_abnormal_predict = modelInstance.ModelPredictForOneplate(model_thinkness_abnormal, test_data_thinkness_abnormal_flag,self.onePlateUpid)
        model_horizon_wave_predict = modelInstance.ModelPredictForOneplate(model_horizon_wave, test_data_horizon_wave_flag,self.onePlateUpid)
        model_left_wave_predict = modelInstance.ModelPredictForOneplate(model_left_wave, test_data_left_wave_flag,self.onePlateUpid)
        model_right_wave_predict = modelInstance.ModelPredictForOneplate(model_right_wave, test_data_right_wave_flag,self.onePlateUpid)
        print('预测完成')

        model_predict_all = {'model_bend_predict':model_bend_predict.tolist(),
                             'model_thinkness_abnormal_predict':model_thinkness_abnormal_predict.tolist(),
                             'model_horizon_wave_predict':model_horizon_wave_predict.tolist(),
                             'model_left_wave_predict':model_left_wave_predict.tolist(),
                             'model_right_wave_predict':model_right_wave_predict.tolist()}

        return model_predict_all