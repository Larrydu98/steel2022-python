'''
dataFilterAfter
'''
import pandas as pd
from flask import json
from flask_restful import Resource, reqparse
import numpy as np
import requests
import pika, traceback
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
class dataFilterAfter:
    '''
    dataFilterAfter
    '''
    def __init__(self,custom_input):
        self.Fu_M_Cc_Output = custom_input['Fu_M_Cc_Output']
        self.Fu_fladcNew = custom_input['Fu_fladcNew']
        self.M_Pg_Output_new = custom_input['M_Pg_Output_new']
        self.M_Hpass_MeasNew = custom_input['M_Hpass_MeasNew']
        self.hpass_post_raw_all = custom_input['hpass_post_raw_all']
        self.Temperature2DNew = custom_input['Temperature2DNew']
        self.TemperatureP12456New = custom_input['TemperatureP12456New']
        self.flag_df = custom_input['flag_df']

    def run(self):
        '''
        run
        '''
        Fu_M_Cc_Output = self.Fu_M_Cc_Output
        Fu_fladcNew = self.Fu_fladcNew
        M_Pg_Output_new = self.M_Pg_Output_new
        M_Hpass_MeasNew = self.M_Hpass_MeasNew
        hpass_post_raw_all = self.hpass_post_raw_all
        Temperature2DNew = self.Temperature2DNew
        TemperatureP12456New = self.TemperatureP12456New
        flag_df = self.flag_df

        # Fu_M_Cc_Output_drop = Fu_M_Cc_Output_drop.drop(['upid_flag'], axis=1)
        print(Fu_M_Cc_Output.slabid)
        print(Fu_fladcNew.slabid)

        Fu_M_Cc_Output_Fu_fladc = pd.merge(Fu_M_Cc_Output, Fu_fladcNew, on='slabid')

        Fu_M_Cc_Output_Fu_fladc_Pg = pd.merge(Fu_M_Cc_Output_Fu_fladc, M_Pg_Output_new,on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas = pd.merge(Fu_M_Cc_Output_Fu_fladc_Pg,M_Hpass_MeasNew, on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D = pd.merge(Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas, Temperature2DNew, on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456 = pd.merge(Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D, TemperatureP12456New,on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost = pd.merge(Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456,hpass_post_raw_all, on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost_Flag = pd.merge(Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost,flag_df, on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost_Flag_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost_Flag.dropna(axis=0, how='any')
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost_Flag_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost_Flag_drop.drop_duplicates('upid', 'last')
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost_Flag_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost_Flag_drop.reset_index()

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost_Flag_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost_Flag_drop.drop(['index', 'slabid'], axis=1)


        ########## 转化为deep and wide的输入格式
        data = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_Temperature2D_TemperatureP12456_MHpassPost_Flag_drop

        # data_x_wide_num = data_x_wide_num
        # data_x_deep1_num = data_x_deep1_num
        # data_x_deep2_num = data_x_deep2_num
        # data_x_deep3_num = data_x_deep3_num
        # data_x_deep4_num = data_x_deep4_num
        # data_x_deep5_num = data_x_deep5_num
        # data_x_deep6_num = data_x_deep6_num

        data_y = data.upid_flag.values
        upid_df = data.upid
        data_x = data.drop(['upid_flag','upid'], axis=1)

        # 划分wide and deep数据
        data_x_wide = data_x.iloc[:, 0:117]

        data_x_deep1 = data_x.iloc[:, 117:357]
        data_x_deep2 = data_x.iloc[:, 357:657]
        data_x_deep3 = data_x.iloc[:, 657:741]
        data_x_deep4 = data_x.iloc[:, 741:1641]
        data_x_deep5 = data_x.iloc[:, 1641:2141]
        data_x_deep6 = data_x.iloc[:, 2141:2211]

        data_x_wide_np = ((data_x_wide - data_x_wide.min()) / (data_x_wide.max() - data_x_wide.min())).values

        data_x_deep1_np = ((data_x_deep1 - data_x_deep1.min()) / (data_x_deep1.max() - data_x_deep1.min())).values

        data_x_deep2_np = ((data_x_deep2 - data_x_deep2.min()) / (data_x_deep2.max() - data_x_deep2.min())).values

        data_x_deep3_np = ((data_x_deep3 - data_x_deep3.min()) / (data_x_deep3.max() - data_x_deep3.min())).values

        data_x_deep4_np = ((data_x_deep4 - data_x_deep4.min()) / (data_x_deep4.max() - data_x_deep4.min())).values

        data_x_deep5_np = ((data_x_deep5 - data_x_deep5.min()) / (data_x_deep5.max() - data_x_deep5.min())).values

        data_x_deep6_np = ((data_x_deep6 - data_x_deep6.min()) / (data_x_deep6.max() - data_x_deep6.min())).values

        # Data Split

        test_size = 0.1

        #######划分测试集与训练集的upid
        upid_train_np = train_test_split(upid_df.values, test_size=test_size, random_state=42)
        upid_test_np = train_test_split(upid_df.values, test_size=test_size, random_state=42)

        x_train_wide, x_test_wide, y_train_raw, y_test_raw = train_test_split(data_x_wide_np, data_y, test_size=test_size, random_state=42)

        data_y_2class = np_utils.to_categorical(data_y)
        y_train, y_test = train_test_split(data_y_2class, test_size=test_size,random_state=42)

        x_train_deep1, x_test_deep1 = train_test_split(data_x_deep1_np, test_size=test_size, random_state=42)
        x_train_deep2, x_test_deep2 = train_test_split(data_x_deep2_np, test_size=test_size, random_state=42)
        x_train_deep3, x_test_deep3 = train_test_split(data_x_deep3_np, test_size=test_size, random_state=42)
        x_train_deep4, x_test_deep4 = train_test_split(data_x_deep4_np, test_size=test_size, random_state=42)
        x_train_deep5, x_test_deep5 = train_test_split(data_x_deep5_np, test_size=test_size, random_state=42)
        x_train_deep6, x_test_deep6 = train_test_split(data_x_deep6_np, test_size=test_size, random_state=42)

        x_train_num = int(len(data_x_wide) * (1 - test_size))
        x_test_num = int(len(data_x_wide) * test_size) + 1

        # no_classes = y_train.shape[1]

        # wide转变输入格式
        x_train_wide = x_train_wide.reshape((x_train_num, 117, 1))
        x_test_wide = x_test_wide.reshape((x_test_num, 117, 1))

        img_H_wide = x_train_wide.shape[1]
        img_W_wide = x_train_wide.shape[2]
        depth_wide = 1

        x_train_wide = x_train_wide.astype('float32')
        x_test_wide = x_test_wide.astype('float32')

        x_train_wide = x_train_wide.reshape(x_train_wide.shape[0], img_H_wide, img_W_wide,depth_wide)
        x_test_wide = x_test_wide.reshape(x_test_wide.shape[0], img_H_wide, img_W_wide, depth_wide)

        # deep1转变输入格式
        x_train_deep1 = x_train_deep1.reshape((x_train_num, 6, 40))
        x_test_deep1 = x_test_deep1.reshape((x_test_num, 6, 40))

        img_H_deep1 = x_train_deep1.shape[1]
        img_W_deep1 = x_train_deep1.shape[2]
        depth_deep1 = 1

        x_train_deep1 = x_train_deep1.astype('float32')
        x_test_deep1 = x_test_deep1.astype('float32')

        x_train_deep1 = x_train_deep1.reshape(x_train_deep1.shape[0], img_H_deep1, img_W_deep1,depth_deep1)
        x_test_deep1 = x_test_deep1.reshape(x_test_deep1.shape[0], img_H_deep1, img_W_deep1,depth_deep1)

        # deep2转变输入格式
        x_train_deep2 = x_train_deep2.reshape((x_train_num, 3, 100))
        x_test_deep2 = x_test_deep2.reshape((x_test_num, 3, 100))

        img_H_deep2 = x_train_deep2.shape[1]
        img_W_deep2 = x_train_deep2.shape[2]
        depth_deep2 = 1

        x_train_deep2 = x_train_deep2.astype('float32')
        x_test_deep2 = x_test_deep2.astype('float32')

        x_train_deep2 = x_train_deep2.reshape(x_train_deep2.shape[0], img_H_deep2, img_W_deep2,depth_deep2)
        x_test_deep2 = x_test_deep2.reshape(x_test_deep2.shape[0], img_H_deep2, img_W_deep2,depth_deep2)

        # deep3转变输入格式
        x_train_deep3 = x_train_deep3.reshape((x_train_num, 12, 7))
        x_test_deep3 = x_test_deep3.reshape((x_test_num, 12, 7))

        img_H_deep3 = x_train_deep3.shape[1]
        img_W_deep3 = x_train_deep3.shape[2]
        depth_deep3 = 1

        x_train_deep3 = x_train_deep3.astype('float32')
        x_test_deep3 = x_test_deep3.astype('float32')

        x_train_deep3 = x_train_deep3.reshape(x_train_deep3.shape[0], img_H_deep3, img_W_deep3,depth_deep3)
        x_test_deep3 = x_test_deep3.reshape(x_test_deep3.shape[0], img_H_deep3, img_W_deep3,depth_deep3)

        # deep4转变输入格式
        x_train_deep4 = x_train_deep4.reshape((x_train_num, 9, 100))
        x_test_deep4 = x_test_deep4.reshape((x_test_num, 9, 100))

        img_H_deep4 = x_train_deep4.shape[1]
        img_W_deep4 = x_train_deep4.shape[2]
        depth_deep4 = 1

        x_train_deep4 = x_train_deep4.astype('float32')
        x_test_deep4 = x_test_deep4.astype('float32')

        x_train_deep4 = x_train_deep4.reshape(x_train_deep4.shape[0], img_H_deep4, img_W_deep4,depth_deep4)
        x_test_deep4 = x_test_deep4.reshape(x_test_deep4.shape[0], img_H_deep4, img_W_deep4,depth_deep4)

        # deep5转变输入格式
        x_train_deep5 = x_train_deep5.reshape((x_train_num, 5, 100))
        x_test_deep5 = x_test_deep5.reshape((x_test_num, 5, 100))

        img_H_deep5 = x_train_deep5.shape[1]
        img_W_deep5 = x_train_deep5.shape[2]
        depth_deep5 = 1

        x_train_deep5 = x_train_deep5.astype('float32')
        x_test_deep5 = x_test_deep5.astype('float32')

        x_train_deep5 = x_train_deep5.reshape(x_train_deep5.shape[0], img_H_deep5, img_W_deep5,depth_deep5)
        x_test_deep5 = x_test_deep5.reshape(x_test_deep5.shape[0], img_H_deep5, img_W_deep5,depth_deep5)

        # deep6转变输入格式
        x_train_deep6 = x_train_deep6.reshape((x_train_num, 10, 7))
        x_test_deep6 = x_test_deep6.reshape((x_test_num, 10, 7))

        img_H_deep6 = x_train_deep6.shape[1]
        img_W_deep6 = x_train_deep6.shape[2]
        depth_deep6 = 1

        x_train_deep6 = x_train_deep6.astype('float32')
        x_test_deep6 = x_test_deep6.astype('float32')

        x_train_deep6 = x_train_deep6.reshape(x_train_deep6.shape[0], img_H_deep6, img_W_deep6,depth_deep6)
        x_test_deep6 = x_test_deep6.reshape(x_test_deep6.shape[0], img_H_deep6, img_W_deep6,depth_deep6)

        print(y_test)

        custom_output = {'x_train_wide': x_train_wide, 'x_test_wide': x_test_wide, 'y_test_raw': y_test_raw, 'y_train': y_train, 'y_test': y_test,
        'x_train_deep1': x_train_deep1,'x_train_deep2': x_train_deep2, 'x_train_deep3': x_train_deep3, 'x_train_deep4': x_train_deep4, 'x_train_deep5': x_train_deep5, 'x_train_deep6': x_train_deep6,
        'x_test_deep1': x_test_deep1, 'x_test_deep2': x_test_deep2, 'x_test_deep3': x_test_deep3, 'x_test_deep4': x_test_deep4, 'x_test_deep5': x_test_deep5, 'x_test_deep6': x_test_deep6,
        'upid_train_np' : upid_train_np , 'upid_test_np' : upid_test_np}

        return custom_output

