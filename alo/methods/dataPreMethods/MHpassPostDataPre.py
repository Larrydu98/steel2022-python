'''
MHpassPostDataPre
'''
# https://scikit-learn.org/stable/modules/svm.html#svr
from flask import json
import pandas as pd
import numpy as np

class MHpassPostDataPre:
    '''
    MHpassPostDataPre
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
        # 道次规划模型出口平直度，厚度后计算值的和进行预处理
        # 数据展开
        upid = pd.DataFrame(custom_input['upid'].tolist(), columns=['upid'])

        contactlength = pd.DataFrame(custom_input['contactlength'].tolist())
        entryflatness = pd.DataFrame(custom_input['entryflatness'].tolist())
        exitflatness = pd.DataFrame(custom_input['exitflatness'].tolist())
        exitprofile = pd.DataFrame(custom_input['exitprofile'].tolist())
        exittemperature = pd.DataFrame(custom_input['exittemperature'].tolist())
        exitthickness = pd.DataFrame(custom_input['exitthickness'].tolist())
        exitwidth = pd.DataFrame(custom_input['exitwidth'].tolist())
        forcecorrection = pd.DataFrame(custom_input['forcecorrection'].tolist())
        rollforcePost = pd.DataFrame(custom_input['rollforcePost'].tolist())
        torquePost = pd.DataFrame(custom_input['torquePost'].tolist())

        # 测量数据拼接
        Mhpass_post_new = pd.concat([upid,contactlength,entryflatness,exitflatness,
                                     exitprofile,exittemperature,exitthickness,exitwidth,
                                     forcecorrection,rollforcePost,torquePost], axis=1)

        # 数据清洗
        Mhpass_post_new_drop = Mhpass_post_new.dropna(axis=0, how='any')
        Mhpass_post_new_drop = Mhpass_post_new_drop.drop_duplicates('upid', 'last')
        Mhpass_post_new_drop = Mhpass_post_new_drop.reset_index()
        Mhpass_post_new_drop = Mhpass_post_new_drop.drop(['index'], axis=1)



        # ####### 转变为模型的输入（deep6）
        # data_x_deep6 = Mhpass_post_new_drop.drop(['upid'], axis=1)

        # data_x_deep6_np = ((data_x_deep6 - data_x_deep6.min()) / (data_x_deep6.max() - data_x_deep6.min())).values

        # # 集成成参数
        # test_size = 0.1
        # x_train_deep6, x_test_deep6 = train_test_split(data_x_deep6_np, test_size=test_size, random_state=42)

        # x_train_num = int(len(data_x_deep6) * (1 - test_size))
        # x_test_num = int(len(data_x_deep6) * test_size) + 1

        # x_train_deep6 = x_train_deep6.reshape((x_train_num, 10, 7))
        # x_test_deep6 = x_test_deep6.reshape((x_test_num, 10, 7))

        # # Data Preperation

        # img_H_deep6 = x_train_deep6.shape[1]
        # img_W_deep6 = x_train_deep6.shape[2]
        # depth_deep6 = 1

        # x_train_deep6 = x_train_deep6.astype('float32')
        # x_test_deep6 = x_test_deep6.astype('float32')

        # x_train_deep6 = x_train_deep6.reshape(x_train_deep6.shape[0], img_H_deep6, img_W_deep6,
        #                                       depth_deep6)
        # x_test_deep6 = x_test_deep6.reshape(x_test_deep6.shape[0], img_H_deep6, img_W_deep6,
        #                                     depth_deep6)

        ##### 输出x_train_wide，x_test_wide，y_train，y_test
        hpass_post_raw_all = Mhpass_post_new_drop

        return hpass_post_raw_all
       





