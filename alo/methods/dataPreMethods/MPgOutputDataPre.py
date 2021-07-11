'''
MPgOutputDataPre
'''
# https://scikit-learn.org/stable/modules/svm.html#svr
from flask import json
import pandas as pd
import numpy as np

class MPgOutputDataPre:
    '''
    MPgOutputDataPre
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
        # 凸度仪测量数据进行预处理
        upid = pd.DataFrame(custom_input['upid'].tolist(), columns=['upid'])

        centerthickness = pd.DataFrame(custom_input['centerthickness'].tolist())
        leftthickness = pd.DataFrame(custom_input['leftthickness'].tolist())
        rightthickness = pd.DataFrame(custom_input['rightthickness'].tolist())

        # 数据清洗
        Pg_Output_all_noupid = pd.concat([centerthickness, leftthickness, rightthickness], axis=1)
        Pg_Output_all_noupid = Pg_Output_all_noupid[(Pg_Output_all_noupid < 100)]
        Pg_Output_all = pd.concat([upid, Pg_Output_all_noupid], axis=1)

        # 数据清洗
        Pg_Output_all_drop = Pg_Output_all.dropna(axis=0, how='any')

        Pg_Output_all_drop = Pg_Output_all_drop.drop_duplicates('upid', 'last')
        Pg_Output_all_drop = Pg_Output_all_drop.reset_index()
        Pg_Output_all_drop = Pg_Output_all_drop.drop(['index'], axis=1)


        # ####### 转变为模型的输入（deep2）
        # data_x_deep2 = Pg_Output_all_drop.drop(['upid'], axis=1)

        # data_x_deep2_np = ((data_x_deep2 - data_x_deep2.min()) / (data_x_deep2.max() - data_x_deep2.min())).values

        # # 集成成参数
        # test_size = 0.1
        # x_train_deep2, x_test_deep2 = train_test_split(data_x_deep2_np, test_size=test_size, random_state=42)

        # x_train_num = int(len(data_x_deep2) * (1 - test_size))
        # x_test_num = int(len(data_x_deep2) * test_size) + 1

        # x_train_deep2 = x_train_deep2.reshape((x_train_num, 3, 100))
        # x_test_deep2 = x_test_deep2.reshape((x_test_num, 3, 100))

        # # Data Preperation

        # img_H_deep2 = x_train_deep2.shape[1]
        # img_W_deep2 = x_train_deep2.shape[2]
        # depth_deep2 = 1

        # x_train_deep2 = x_train_deep2.astype('float32')
        # x_test_deep2 = x_test_deep2.astype('float32')

        # x_train_deep2 = x_train_deep2.reshape(x_train_deep2.shape[0], img_H_deep2, img_W_deep2,depth_deep2)
        # x_test_deep2 = x_test_deep2.reshape(x_test_deep2.shape[0], img_H_deep2, img_W_deep2,depth_deep2)

        ##### 输出x_train_wide，x_test_wide，y_train，y_test
        M_Pg_Output= Pg_Output_all_drop

        return M_Pg_Output
       





