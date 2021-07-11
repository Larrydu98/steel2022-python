'''
FuFladcDataPre
'''

from flask import json
import pandas as pd
import numpy as np

class FuFladcDataPre:
    '''
    FuFladcDataPre
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
        # 炉温数据进行预处理
        slabid = pd.DataFrame(custom_input['slabid'].tolist(), columns=['slabid'])

        temp_seg_ul_1 = pd.DataFrame(custom_input['temp_seg_ul_1'].tolist())
        temp_seg_ul_2 = pd.DataFrame(custom_input['temp_seg_ul_2'].tolist())
        temp_seg_dl_s = pd.DataFrame(custom_input['temp_seg_dl_s'].tolist())
        # temp_seg_dl_b = pd.DataFrame(Fu_fladc['temp_seg_dl_b'].tolist())
        temp_seg_ur_1 = pd.DataFrame(custom_input['temp_seg_ur_1'].tolist())
        temp_seg_ur_2 = pd.DataFrame(custom_input['temp_seg_ur_2'].tolist())
        temp_seg_dr_s = pd.DataFrame(custom_input['temp_seg_dr_s'].tolist())
        # temp_seg_dr_b = pd.DataFrame(Fu_fladc['temp_seg_dr_b'].tolist())

        # 炉温的展开数据合并
        Fu_fladc_new = pd.concat([slabid,
                                  temp_seg_ul_1,
                                  temp_seg_ul_2,
                                  temp_seg_dl_s,
                                  #                       temp_seg_dl_b,
                                  temp_seg_ur_1,
                                  temp_seg_ur_2,
                                  temp_seg_dr_s
                                  #                       temp_seg_dr_b
                                  ], axis=1)

        # 数据清理
        Fu_fladc_new_drop = Fu_fladc_new.dropna(axis=0, how='any')
        Fu_fladc_new_drop = Fu_fladc_new_drop.drop_duplicates('slabid','last')
        Fu_fladc_new_drop = Fu_fladc_new_drop.reset_index()
        Fu_fladc_new_drop = Fu_fladc_new_drop.drop(['index'], axis=1)

        # ####### 转变为模型的输入（deep1）
        # data_x_deep1 = Fu_fladc_new_drop.drop(['slabid'], axis=1)

        # data_x_deep1_np = ((data_x_deep1 - data_x_deep1.min()) / (data_x_deep1.max() - data_x_deep1.min())).values

        # # 集成成参数
        # test_size = 0.1
        # x_train_deep1, x_test_deep1 = train_test_split(data_x_deep1_np, test_size=test_size, random_state=42)

        # x_train_num = int(len(data_x_deep1) * (1 - test_size))
        # x_test_num = int(len(data_x_deep1) * test_size) + 1

        # x_train_deep1 = x_train_deep1.reshape((x_train_num, 6, 40))
        # x_test_deep1 = x_test_deep1.reshape((x_test_num, 6, 40))

        # # Data Preperation

        # img_H_deep1 = x_train_deep1.shape[1]
        # img_W_deep1 = x_train_deep1.shape[2]
        # depth_deep1 = 1

        # x_train_deep1 = x_train_deep1.astype('float32')
        # x_test_deep1 = x_test_deep1.astype('float32')

        # x_train_deep1 = x_train_deep1.reshape(x_train_deep1.shape[0], img_H_deep1, img_W_deep1,depth_deep1)
        # x_test_deep1 = x_test_deep1.reshape(x_test_deep1.shape[0], img_H_deep1, img_W_deep1,depth_deep1)

        ##### 输出x_train_wide，x_test_wide，y_train，y_test
        Fu_fladc = Fu_fladc_new_drop

        return Fu_fladc
       





