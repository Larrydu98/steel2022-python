'''
SequenceDataPre
'''
# https://scikit-learn.org/stable/modules/svm.html#svr
from flask import json
import pandas as pd
import numpy as np

class SequenceDataPre:
    '''
    SequenceDataPre
    '''
    def __init__(self):
        self.paramsIllegal = False
        try:
            print('这个方法暂时没有参数')
        except Exception:
            self.paramsIllegal = True

    def general_call(self, custom_input):
        '''
        general_call
        '''
        # 单维序列数据进行预处理
        Fu_M_Cc_Output_filter = custom_input[['ave_temp_1', 'ave_temp_2', 'ave_temp_dis', 'ave_temp_entry_1',
        'ave_temp_entry_2', 'ave_temp_entry_pre', 'ave_temp_entry_soak',
        'ave_temp_pre', 'ave_temp_soak', 'avg_fct', 'avg_p1', 'avg_p2',
        'avg_p5', 'avg_sct', 'botbrplatecountfm', 'botbrplatecountrm',
        'botwrplatecountfm', 'botwrplatecountrm', 'center_temp_dis',
        'center_temp_entry_1', 'center_temp_entry_2',
        'center_temp_entry_pre', 'center_temp_entry_soak',
        'charging_temp_act', 'crownbody', 'crownhead', 'crowntail',
        'crowntotal', 'devcrownbody', 'devcrownhead', 'devcrowntail',
        'devcrowntotal', 'devfinishtempbody', 'devfinishtemphead',
        'devfinishtemptail', 'devfinishtemptotal', 'devwedgebody',
        'devwedgehead', 'devwedgetail', 'devwedgetotal', 'finishtempbody',
        'finishtemphead', 'finishtemptail', 'finishtemptotal', 'max_fct',
        'max_p1', 'max_p2', 'max_p5', 'max_sct', 'meas_temp_0',
        'meas_temp_1', 'meas_temp_10', 'meas_temp_11', 'meas_temp_12',
        'meas_temp_13', 'meas_temp_14', 'meas_temp_15', 'meas_temp_16',
        'meas_temp_17', 'meas_temp_18', 'meas_temp_19', 'meas_temp_2',
        'meas_temp_3', 'meas_temp_4', 'meas_temp_5', 'meas_temp_6',
        'meas_temp_7', 'meas_temp_8', 'meas_temp_9', 'min_fct', 'min_p1',
        'min_p2', 'min_p5', 'min_sct', 'pass',
        'skid_temp_dis', 'skid_temp_entry_1', 'skid_temp_entry_2',
        'skid_temp_entry_pre', 'skid_temp_entry_soak', 'slab_length',
        'slab_thickness', 'slab_weight_act', 'slab_width', 'slabid',
        'staying_time_1', 'staying_time_2', 'staying_time_pre',
        'staying_time_soak', 'std_fct', 'std_p1', 'std_p2', 'std_p5',
        'std_sct', 'sur_temp_dis', 'sur_temp_entry_1', 'sur_temp_entry_2',
        'sur_temp_entry_pre', 'sur_temp_entry_soak', 't_0', 't_1', 't_2',
        't_3', 't_4', 't_5', 't_6', 'temp_uniformity_dis',
        'temp_uniformity_entry_1', 'temp_uniformity_entry_2',
        'temp_uniformity_entry_pre', 'temp_uniformity_entry_soak',
        'tgtplatelength2', 'tgtplatethickness2', 'tgtwidth', 'upid',
        'wedgebody', 'wedgehead', 'wedgetail', 'wedgetotal']]

        Fu_M_Cc_Output_drop = Fu_M_Cc_Output_filter.dropna(axis=0, how='any')
        Fu_M_Cc_Output_drop = Fu_M_Cc_Output_drop.drop_duplicates('upid','last')
        Fu_M_Cc_Output_drop = Fu_M_Cc_Output_drop.reset_index()
        Fu_M_Cc_Output_drop = Fu_M_Cc_Output_drop.drop(['index'], axis=1)
        # ###### 转变为模型的输入（wide）
        # data_y = Fu_M_Cc_Output_drop.upid_flag.values
        # data_x_wide = Fu_M_Cc_Output_drop.drop(['upid','slabid','upid_flag'],axis=1)

        # data_x_wide_np = ((data_x_wide - data_x_wide.min()) / (data_x_wide.max() - data_x_wide.min())).values

        # # 集成成参数
        # test_size=0.1
        # x_train_wide, x_test_wide, y_train, y_test = train_test_split(data_x_wide_np, data_y, test_size=test_size, random_state=42)

        # y_test_raw = y_test

        # x_train_num = int(len(data_x_wide)*(1-test_size))
        # x_test_num = int(len(data_x_wide)*test_size)+1

        # # import keras包
        # y_train = np_utils.to_categorical(y_train)
        # y_test = np_utils.to_categorical(y_test)

        # x_train_wide = x_train_wide.reshape((x_train_num, len(x_train_wide[0]), 1))
        # x_test_wide = x_test_wide.reshape((x_test_num, 117, 1))

        # img_H_wide = x_train_wide.shape[1]
        # img_W_wide = x_train_wide.shape[2]
        # depth_wide = 1

        # x_train_wide = x_train_wide.astype('float32')
        # x_test_wide = x_test_wide.astype('float32')

        # x_train_wide = x_train_wide.reshape(x_train_wide.shape[0], img_H_wide, img_W_wide,depth_wide)
        # x_test_wide = x_test_wide.reshape(x_test_wide.shape[0], img_H_wide, img_W_wide, depth_wide)
        # ###### 输出x_train_wide，x_test_wide，y_train，y_test
        Fu_M_Cc_Output = Fu_M_Cc_Output_drop

        return Fu_M_Cc_Output
       





