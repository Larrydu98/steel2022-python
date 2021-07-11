'''
dataFilterCsvAfter
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
class dataFilterCsvAfter:
    '''
    dataFilterCsvAfter
    '''
    def __init__(self):
        print('生成实例')

    def run(self):
        '''
        run
        '''
        path = os.getcwd()
        data = pd.read_csv(path + staticPath.data_all_w)
        # data = pd.read_csv(path + staticPath.data_all_l)
        ########## 转化为deep and wide的输入格式

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
        upid_train_np,upid_test_np = train_test_split(upid_df.values, test_size=test_size, random_state=42)

        x_train_wide, x_test_wide, y_train_raw, y_test_raw = train_test_split(data_x_wide_np,data_y,test_size=test_size,random_state=42)

        data_y_2class = np_utils.to_categorical(data_y)
        y_train, y_test = train_test_split(data_y_2class, test_size=test_size, random_state=42)

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

        custom_output = {'x_train_wide': x_train_wide, 'x_test_wide': x_test_wide, 'y_test_raw': y_test_raw, 'y_train': y_train, 'y_test': y_test,
        'x_train_deep1': x_train_deep1,'x_train_deep2': x_train_deep2, 'x_train_deep3': x_train_deep3, 'x_train_deep4': x_train_deep4, 'x_train_deep5': x_train_deep5, 'x_train_deep6': x_train_deep6,
        'x_test_deep1': x_test_deep1, 'x_test_deep2': x_test_deep2, 'x_test_deep3': x_test_deep3, 'x_test_deep4': x_test_deep4, 'x_test_deep5': x_test_deep5, 'x_test_deep6': x_test_deep6,
        'upid_train_np' : upid_train_np , 'upid_test_np' : upid_test_np}

        return custom_output

class dataFilterCsvAfterAll:
    '''
    dataFilterCsvAfterForSomePlate
    '''
    def __init__(self):
        print('生成实例')

    def run(self):
        '''
        run
        '''
        path = os.getcwd()
        data = pd.read_csv(path + staticPath.data_all_w)
        # data = pd.read_csv(path + staticPath.data_all_l)
        ########## 转化为deep and wide的输入格式

        # data_x_wide_num = data_x_wide_num
        # data_x_deep1_num = data_x_deep1_num
        # data_x_deep2_num = data_x_deep2_num
        # data_x_deep3_num = data_x_deep3_num
        # data_x_deep4_num = data_x_deep4_num
        # data_x_deep5_num = data_x_deep5_num
        # data_x_deep6_num = data_x_deep6_num

        data_y = data.upid_flag.values
        upid_np = data.upid.values
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

        x_num = int(len(data_x_wide))

        data_y_2class = np_utils.to_categorical(data_y)

        # wide转变输入格式
        x_train_wide = data_x_wide_np.reshape((x_num, 117, 1))

        img_H_wide = x_train_wide.shape[1]
        img_W_wide = x_train_wide.shape[2]
        depth_wide = 1

        x_train_wide = x_train_wide.astype('float32')
        x_train_wide = x_train_wide.reshape(x_train_wide.shape[0], img_H_wide, img_W_wide,depth_wide)

        # deep1转变输入格式
        x_train_deep1 = data_x_deep1_np.reshape((x_num, 6, 40))

        img_H_deep1 = x_train_deep1.shape[1]
        img_W_deep1 = x_train_deep1.shape[2]
        depth_deep1 = 1

        x_train_deep1 = x_train_deep1.astype('float32')
        x_train_deep1 = x_train_deep1.reshape(x_train_deep1.shape[0], img_H_deep1, img_W_deep1,depth_deep1)

        # deep2转变输入格式
        x_train_deep2 = data_x_deep2_np.reshape((x_num, 3, 100))

        img_H_deep2 = x_train_deep2.shape[1]
        img_W_deep2 = x_train_deep2.shape[2]
        depth_deep2 = 1

        x_train_deep2 = x_train_deep2.astype('float32')
        x_train_deep2 = x_train_deep2.reshape(x_train_deep2.shape[0], img_H_deep2, img_W_deep2,depth_deep2)

        # deep3转变输入格式
        x_train_deep3 = data_x_deep3_np.reshape((x_num, 12, 7))

        img_H_deep3 = x_train_deep3.shape[1]
        img_W_deep3 = x_train_deep3.shape[2]
        depth_deep3 = 1

        x_train_deep3 = x_train_deep3.astype('float32')
        x_train_deep3 = x_train_deep3.reshape(x_train_deep3.shape[0], img_H_deep3, img_W_deep3,depth_deep3)

        # deep4转变输入格式
        x_train_deep4 = data_x_deep4_np.reshape((x_num, 9, 100))

        img_H_deep4 = x_train_deep4.shape[1]
        img_W_deep4 = x_train_deep4.shape[2]
        depth_deep4 = 1

        x_train_deep4 = x_train_deep4.astype('float32')
        x_train_deep4 = x_train_deep4.reshape(x_train_deep4.shape[0], img_H_deep4, img_W_deep4,depth_deep4)

        # deep5转变输入格式
        x_train_deep5 = data_x_deep5_np.reshape((x_num, 5, 100))

        img_H_deep5 = x_train_deep5.shape[1]
        img_W_deep5 = x_train_deep5.shape[2]
        depth_deep5 = 1

        x_train_deep5 = x_train_deep5.astype('float32')
        x_train_deep5 = x_train_deep5.reshape(x_train_deep5.shape[0], img_H_deep5, img_W_deep5,depth_deep5)

        # deep6转变输入格式
        x_train_deep6 = data_x_deep6_np.reshape((x_num, 10, 7))

        img_H_deep6 = x_train_deep6.shape[1]
        img_W_deep6 = x_train_deep6.shape[2]
        depth_deep6 = 1

        x_train_deep6 = x_train_deep6.astype('float32')
        x_train_deep6 = x_train_deep6.reshape(x_train_deep6.shape[0], img_H_deep6, img_W_deep6,depth_deep6)

        custom_output = {'x_wide': x_train_wide, 'y_data2class': data_y_2class, 'y_data1class': data_y,
                         'x_deep1': x_train_deep1,'x_deep2': x_train_deep2, 'x_deep3': x_train_deep3, 'x_deep4': x_train_deep4,
                         'x_deep5': x_train_deep5, 'x_deep6': x_train_deep6,
                         'upid_np' : upid_np }

        return custom_output


class getDataFilterCsvAfterTimeByUpid:
    '''
    getDataFilterCsvAfterTimeByUpid
    '''
    def __init__(self):
        print('生成实例')

    def run(self,upid,upidBeforeTimeDay):
        '''
        run
        '''

        ########## 转化为deep and wide的输入格式

        # startTime = datetime.datetime.strptime(startTime, '%Y-%m-%d %H:%M:%S')
        # endTime = datetime.datetime.strptime(endTime, '%Y-%m-%d %H:%M:%S')
        # data['toc'] = pd.to_datetime(data['toc'], format='%Y-%m-%d %H:%M:%S')
        # data = data[(data['toc'] > startTime) & (data['toc'] < endTime)]

        api = 'http://java-serve:8080/web-ssm/FuMCcTest1/selectFuMCcTest1/2018-9-1 00:00:00/2018-9-30 24:00:00/'+upid+'/'
        Fu_M_Cc_Output = requests.get(api)
        Fu_M_Cc_Output = pd.DataFrame(json.loads(Fu_M_Cc_Output.content))

        endTime_str = Fu_M_Cc_Output.toc[0]

        endTime = datetime.datetime.strptime(endTime_str, '%Y-%m-%d %H:%M:%S')
        startTime_str = str(endTime + datetime.timedelta(days=-upidBeforeTimeDay))

        return startTime_str,endTime_str


class dataFilterCsvAfterByTime:
    '''
    dataFilterCsvAfterByTime
    '''
    def __init__(self):
        print('生成实例')

    def run(self,startTime,endTime):
        '''
        run
        '''
        path = os.getcwd()
        data = pd.read_csv(path + staticPath.data_all_w)
        ########## 转化为deep and wide的输入格式

        # startTime = datetime.datetime.strptime(startTime, '%Y-%m-%d %H:%M:%S')
        # endTime = datetime.datetime.strptime(endTime, '%Y-%m-%d %H:%M:%S')
        # data['toc'] = pd.to_datetime(data['toc'], format='%Y-%m-%d %H:%M:%S')
        # data = data[(data['toc'] > startTime) & (data['toc'] < endTime)]

        api = 'http://java-serve:8080/web-ssm/FuMCcTest1/selectFuMCcTest1/'+startTime+'/'+endTime+'/all/'
        Fu_M_Cc_Output = requests.get(api)
        Fu_M_Cc_Output = pd.DataFrame(json.loads(Fu_M_Cc_Output.content))
        Fu_M_Cc_Output[['upid']] = Fu_M_Cc_Output[['upid']].astype('int64')
        upid_toc = Fu_M_Cc_Output[['upid', 'toc']]

        data_ByTime = pd.merge(data, upid_toc, on='upid')
        data_ByTime = data_ByTime.dropna(axis=0, how='any').drop_duplicates('upid', 'last').reset_index().drop(['index'], axis=1)

        upid_byTime = data_ByTime[['upid']]

        if data_ByTime.empty:
            print('该段时间不存在钢板数据')
        else:

            InputData = dataFilterCsvAfterAll().run()

            upid_all = pd.DataFrame(InputData['upid_np'],columns=['upid'])

            upid_index = upid_all[upid_all["upid"].isin(upid_byTime.values.reshape(len(upid_byTime)).tolist())].index.tolist()

            x_wide = InputData['x_wide']
            x_deep1 = InputData['x_deep1']
            x_deep2 = InputData['x_deep2']
            x_deep3 = InputData['x_deep3']
            x_deep4 = InputData['x_deep4']
            x_deep5 = InputData['x_deep5']
            x_deep6 = InputData['x_deep6']
            y_data1class = InputData['y_data1class']
            y_data2class = InputData['y_data2class']

            custom_output = {'x_wide': x_wide[upid_index], 'y_data2class': y_data2class[upid_index], 'y_data1class': y_data1class[upid_index],
                             'x_deep1': x_deep1[upid_index],'x_deep2': x_deep2[upid_index], 'x_deep3': x_deep3[upid_index], 'x_deep4': x_deep4[upid_index],
                             'x_deep5': x_deep5[upid_index], 'x_deep6': x_deep6[upid_index],
                             'upid_np' : upid_byTime.values.reshape(len(upid_byTime)) }

        return custom_output
