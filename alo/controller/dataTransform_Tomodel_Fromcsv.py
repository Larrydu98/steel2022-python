'''
dataTransform_Tomodel_Fromcsv
'''
import pandas as pd
from flask import json
from flask_restful import Resource, reqparse
import numpy as np
import requests
import pika, traceback
import datetime
import math
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import os
from ..staticPath import staticPath


class dataTransform_OneAbnormal_Fromcsv:
    '''
    dataFilterCsvAfter
    '''
    def __init__(self,data_train_start,data_train_end,data_test_start,data_test_end):
        path = os.getcwd()
        ProcessData = pd.read_csv(path + staticPath.Process_data)
        ProcessData['upid'] = ProcessData['upid'].astype('str')
        self.ProcessData = ProcessData

        self.data_train_start = data_train_start
        self.data_train_end = data_train_end
        self.data_test_start = data_test_start
        self.data_test_end = data_test_end
        print('生成实例')

    def run(self,OneAbnormal_flag):
        '''
        run
        '''
        path = os.getcwd()

        ##########读取数据
        # ProcessData = pd.read_csv(path + staticPath.Process_data)
        flagData = pd.read_csv(path + staticPath.flag_data)

        #########为钢板过程数据附上一种板形异常标签
        all_OneAbnormal_flag_df = flagData[['upid', OneAbnormal_flag]]
        all_OneAbnormal_flag_df.columns = ['upid', 'upid_flag']
        all_OneAbnormal_flag_new_df = all_OneAbnormal_flag_df[all_OneAbnormal_flag_df.upid_flag == 1].replace(to_replace=1, value=0).append(all_OneAbnormal_flag_df[all_OneAbnormal_flag_df.upid_flag == 0].replace(to_replace=0,value=1)).sort_values(by='upid').reset_index().drop(['index'], axis=1)
        all_OneAbnormal_flag_new_df['upid'] = all_OneAbnormal_flag_new_df['upid'].astype('str')
        data_new_OneAbnormal_flag = pd.merge(self.ProcessData.drop('upid_flag', axis=1), all_OneAbnormal_flag_new_df,on='upid')
        all_x_data_ylist = []
        for i in range(50, len(data_new_OneAbnormal_flag)):
            x_data_ylist = data_new_OneAbnormal_flag['upid_flag'][i - 50:i].values
            all_x_data_ylist.append(x_data_ylist)
        all_x_data_y_df = pd.DataFrame(all_x_data_ylist)
        data_y_upid = pd.DataFrame(data_new_OneAbnormal_flag.iloc[50:len(data_new_OneAbnormal_flag)].reset_index().drop(['index'],axis=1).upid)
        all_x_data_y_upid_df = pd.concat([data_y_upid, all_x_data_y_df], axis=1)
        data_addY_new = pd.merge(all_x_data_y_upid_df, data_new_OneAbnormal_flag)

        #选择训练集与测试集
        data_train = data_addY_new[(data_addY_new.upid > self.data_train_start) & (data_addY_new.upid < self.data_train_end)]
        data_test = data_addY_new[(data_addY_new.upid > self.data_test_start) & (data_addY_new.upid < self.data_test_end)]
        x_train_num = len(data_train)
        x_test_num = len(data_test)
        data_all = data_train.append(data_test)

        data_y = data_all.upid_flag.values
        data_upid = data_all.upid.values
        data_x = data_all.drop(['upid_flag', 'upid'], axis=1)

        #归一化
        data_x_np = ((data_x - data_x.min()) / (data_x.max() - data_x.min())).fillna(0).values
        data_train_np = data_x_np[0:len(data_train)]
        y_train = data_y[0:len(data_train)]
        data_train_upid = data_upid[0:len(data_train)]

        #打乱训练数据顺序
        data_train_np, a, y_train, b, data_train_upid, c = train_test_split(data_train_np, y_train, data_train_upid, test_size=0, random_state=50)
        data_test_np = data_x_np[len(data_train):len(data_all)]
        y_test = data_y[len(data_train):len(data_all)]
        data_test_upid = data_upid[len(data_train):len(data_all)]

        #划分数据块
        x_train_yList = data_train_np[:, 0:50]
        x_train_wide = data_train_np[:, 50:50 + 570]
        x_train_deep1 = data_train_np[:, 50 + 570:50 + 570 + 1566]
        x_train_deep2 = data_train_np[:, 50 + 570 + 1566:50 + 570 + 1566 + 2856]
        x_train_deep3 = data_train_np[:, 50 + 570 + 1566 + 2856:50 + 570 + 1566 + 2856 + 1904]
        x_train_deep4 = data_train_np[:,50 + 570 + 1566 + 2856 + 1904:50 + 570 + 1566 + 2856 + 1904 + 1683]
        x_train_deep5 = data_train_np[:,50 + 570 + 1566 + 2856 + 1904 + 1683:50 + 570 + 1566 + 2856 + 1904 + 1683 + 1325]

        x_test_yList = data_test_np[:, 0:50]
        x_test_wide = data_test_np[:, 50:50 + 570]
        x_test_deep1 = data_test_np[:, 50 + 570:50 + 570 + 1566]
        x_test_deep2 = data_test_np[:, 50 + 570 + 1566:50 + 570 + 1566 + 2856]
        x_test_deep3 = data_test_np[:, 50 + 570 + 1566 + 2856:50 + 570 + 1566 + 2856 + 1904]
        x_test_deep4 = data_test_np[:,50 + 570 + 1566 + 2856 + 1904:50 + 570 + 1566 + 2856 + 1904 + 1683]
        x_test_deep5 = data_test_np[:,50 + 570 + 1566 + 2856 + 1904 + 1683:50 + 570 + 1566 + 2856 + 1904 + 1683 + 1325]

        y_raw_train = y_train
        y_raw_test = y_test

        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        ############train转变输入格式
        # yList转变输入格式
        x_train_yList = x_train_yList.reshape((x_train_num, 50, 1, 1))
        x_train_yList = x_train_yList.astype('float32')

        # wide转变输入格式
        x_train_wide = x_train_wide.reshape((x_train_num, 570, 1, 1))
        x_train_wide = x_train_wide.astype('float32')

        # deep1转变输入格式
        x_train_deep1 = x_train_deep1.reshape((x_train_num, 6, 261, 1))
        x_train_deep1 = x_train_deep1.astype('float32')

        # deep2转变输入格式
        x_train_deep2 = x_train_deep2.reshape((x_train_num, 34, 12, 7, 1))
        x_train_deep2 = x_train_deep2.astype('float32')

        # deep3转变输入格式
        x_train_deep3 = x_train_deep3.reshape((x_train_num, 34, 8, 7, 1))
        x_train_deep3 = x_train_deep3.astype('float32')

        # deep4转变输入格式
        x_train_deep4 = x_train_deep4.reshape((x_train_num), 9, 187, 1)
        x_train_deep4 = x_train_deep4.astype('float32')

        # deep5转变输入格式
        x_train_deep5 = x_train_deep5.reshape((x_train_num), 5, 265, 1)
        x_train_deep5 = x_train_deep5.astype('float32')

        ###############test转变输入格式
        # yList转变输入格式
        x_test_yList = x_test_yList.reshape((x_test_num, 50, 1, 1))
        x_test_yList = x_test_yList.astype('float32')

        # wide转变输入格式
        x_test_wide = x_test_wide.reshape((x_test_num, 570, 1, 1))
        x_test_wide = x_test_wide.astype('float32')

        # deep1转变输入格式
        x_test_deep1 = x_test_deep1.reshape((x_test_num, 6, 261, 1))
        x_test_deep1 = x_test_deep1.astype('float32')

        # deep2转变输入格式
        x_test_deep2 = x_test_deep2.reshape((x_test_num, 34, 12, 7, 1))
        x_test_deep2 = x_test_deep2.astype('float32')

        # deep3转变输入格式
        x_test_deep3 = x_test_deep3.reshape((x_test_num, 34, 8, 7, 1))
        x_test_deep3 = x_test_deep3.astype('float32')

        # deep4转变输入格式
        x_test_deep4 = x_test_deep4.reshape((x_test_num, 9, 187, 1))
        x_test_deep4 = x_test_deep4.astype('float32')

        # deep5转变输入格式
        x_test_deep5 = x_test_deep5.reshape((x_test_num, 5, 265, 1))
        x_test_deep5 = x_test_deep5.astype('float32')

        train_data_all = {'data_train_upid' : data_train_upid ,'y_raw_train': y_raw_train, 'y_train': y_train,
                          'x_train_yList': x_train_yList,'x_train_wide': x_train_wide,
                          'x_train_deep1': x_train_deep1,'x_train_deep2': x_train_deep2, 'x_train_deep3': x_train_deep3, 'x_train_deep4': x_train_deep4, 'x_train_deep5': x_train_deep5}

        test_data_all = {'data_test_upid' : data_test_upid, 'y_raw_test': y_raw_test, 'y_test': y_test,
                         'x_test_yList': x_test_yList, 'x_test_wide': x_test_wide,
                         'x_test_deep1': x_test_deep1, 'x_test_deep2': x_test_deep2, 'x_test_deep3': x_test_deep3, 'x_test_deep4': x_test_deep4, 'x_test_deep5': x_test_deep5 }

        return train_data_all,test_data_all