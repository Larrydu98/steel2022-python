'''
deepModelPredictController
'''
import numpy as np
from flask import json
from keras.models import model_from_json
from keras.models import load_model
import os
from ..staticPath import staticPath
from keras import backend as K


class deepModelPredict:
    '''
    deepModelPredict
    '''

    def __init__(self):
        self.idIllegal = False
        print('生成实例')

    # 创建模型结构实例
    def createModelStructure(self):
        '''
        createModelStructure
        '''
        path = os.getcwd()
        print('load model Structure...')
        json_string = open(path + staticPath.model_Structure).read()
        model_Structure = model_from_json(json_string)
        print('load done.')
        return model_Structure

    # 加载模型权重实例
    def createModelWeights(self,model_Structure,OneAbnormal_flag):
        '''
        createModelWeights
        '''
        path = os.getcwd()
        print('load model Weights...')
        # json_string = open(path + staticPath.model_Structure).read()
        # model_Structure = model_from_json(json_string)

        model = model_Structure
        if OneAbnormal_flag=='bend_flag':
            model.load_weights(path + staticPath.model_bend_weights)
        if OneAbnormal_flag=='thinkness_abnormal_flag':
            model.load_weights(path + staticPath.model_thinkness_abnormal_weights)
        if OneAbnormal_flag=='horizon_wave_flag':
            model.load_weights(path + staticPath.model_horizon_wave_weights)
        if OneAbnormal_flag=='left_wave_flag':
            model.load_weights(path + staticPath.model_left_wave_weights)
        if OneAbnormal_flag=='right_wave_flag':
            model.load_weights(path + staticPath.model_right_wave_weights)

        model.compile(
            optimizer='Adadelta',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print('load done.')
        return model

    def ModelPredictForOneplate(self, model, test_data, onePlateUpid):
        '''
        ModelPredictForOneplate
        '''
        path = os.getcwd()
        x_test_yList = np.array(test_data['x_test_yList'])
        x_test_wide = np.array(test_data['x_test_wide'])
        x_test_deep1 = np.array(test_data['x_test_deep1'])
        x_test_deep2 = np.array(test_data['x_test_deep2'])
        x_test_deep3 = np.array(test_data['x_test_deep3'])
        x_test_deep4 = np.array(test_data['x_test_deep4'])
        x_test_deep5 = np.array(test_data['x_test_deep5'])

        y_test = np.array(test_data['y_test'])
        y_raw_test = np.array(test_data['y_raw_test'])

        data_test_upid = test_data['data_test_upid']

        if len(np.where(data_test_upid == onePlateUpid)[0]) == 0:
            return '不存在的upid的数据'

        if len(np.where(data_test_upid == onePlateUpid)[0]) != 0:
            test_index = np.where(data_test_upid == onePlateUpid)[0][0]

            model_predict = model.predict([x_test_yList[test_index:test_index + 1], x_test_wide[test_index:test_index + 1],
                                           x_test_deep1[test_index:test_index + 1], x_test_deep2[test_index:test_index + 1],
                                           x_test_deep3[test_index:test_index + 1], x_test_deep4[test_index:test_index + 1],
                                           x_test_deep5[test_index:test_index + 1]])


        return model_predict[0]