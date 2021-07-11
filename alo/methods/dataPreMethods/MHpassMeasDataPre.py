'''
MHpassMeasDataPre
'''
# https://scikit-learn.org/stable/modules/svm.html#svr
from flask import json
import pandas as pd
import numpy as np

class MHpassMeasDataPre:
    '''
    MHpassMeasDataPre
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
        #道次规划模型轧制力，弯辊力测量数据的和进行预处理
        # 数据展开
        upid = pd.DataFrame(custom_input['upid'].tolist(), columns=['upid'])

        bendingforce = pd.DataFrame(custom_input['bendingforce'].tolist())
        bendingforcebot = pd.DataFrame(custom_input['bendingforcebot'].tolist())
        bendingforcetop = pd.DataFrame(custom_input['bendingforcetop'].tolist())
        rollforce = pd.DataFrame(custom_input['rollforce'].tolist())
        rollforceds = pd.DataFrame(custom_input['rollforceds'].tolist())
        rollforceos = pd.DataFrame(custom_input['rollforceos'].tolist())
        screwdown = pd.DataFrame(custom_input['screwdown'].tolist())
        shiftpos = pd.DataFrame(custom_input['shiftpos'].tolist())
        speed = pd.DataFrame(custom_input['speed'].tolist())
        torque = pd.DataFrame(custom_input['torque'].tolist())
        torquebot = pd.DataFrame(custom_input['torquebot'].tolist())
        torquetop = pd.DataFrame(custom_input['torquetop'].tolist())

        # 测量数据拼接
        Mhpass_Meas_new = pd.concat([upid, bendingforce, bendingforcebot, bendingforcetop,rollforce,rollforceds, 
                                        rollforceos, screwdown, shiftpos, speed, torque, torquebot, torquetop], axis=1)

        # 数据清洗
        Mhpass_Meas_new_drop = Mhpass_Meas_new.dropna(axis=0, how='any')
        Mhpass_Meas_new_drop = Mhpass_Meas_new_drop.drop_duplicates('upid', 'last')
        Mhpass_Meas_new_drop = Mhpass_Meas_new_drop.reset_index()
        Mhpass_Meas_new_drop = Mhpass_Meas_new_drop.drop(['index'], axis=1)


        # ####### 转变为模型的输入（deep3）
        # data_x_deep3 = Mhpass_Meas_new_drop.drop(['upid'], axis=1)

        # data_x_deep3_np = ((data_x_deep3 - data_x_deep3.min()) / (data_x_deep3.max() - data_x_deep3.min())).values

        # # 集成成参数
        # test_size = 0.1
        # x_train_deep3, x_test_deep3 = train_test_split(data_x_deep3_np, test_size=test_size, random_state=42)

        # x_train_num = int(len(data_x_deep3) * (1 - test_size))
        # x_test_num = int(len(data_x_deep3) * test_size) + 1

        # x_train_deep3 = x_train_deep3.reshape((x_train_num, 12, 7))
        # x_test_deep3 = x_test_deep3.reshape((x_test_num, 12, 7))

        # # Data Preperation

        # img_H_deep3 = x_train_deep3.shape[1]
        # img_W_deep3 = x_train_deep3.shape[2]
        # depth_deep3 = 1

        # x_train_deep3 = x_train_deep3.astype('float32')
        # x_test_deep3 = x_test_deep3.astype('float32')

        # x_train_deep3 = x_train_deep3.reshape(x_train_deep3.shape[0], img_H_deep3, img_W_deep3,
        #                                       depth_deep3)
        # x_test_deep3 = x_test_deep3.reshape(x_test_deep3.shape[0], img_H_deep3, img_W_deep3,
        #                                     depth_deep3)

        ##### 输出x_train_wide，x_test_wide，y_train，y_test
        M_Hpass_MeasNew = Mhpass_Meas_new_drop

        return M_Hpass_MeasNew
       





