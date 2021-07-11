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
        filePath1 = "/usr/src/data/mother_fqc_before/13.csv"
        # "/usr/src/data/mother_fqc_before/" + upid + ".csv"
        # filePath1 = r'/alo/data/wide_117_data_flag_20190831.csv'
        # filePath2 = r'D:\document\BAOGANGDATA\steelspec_data20180902.csv'
        filePath2 = "/usr/src/data/mother_fqc_before/12.csv"
        # filePath2 = r'/alo/data/steelspec_data20180902.csv'

        parser = reqparse.RequestParser(trim=True, bundle_errors=True)
        wideData_dir = pd.read_csv("/usr/src/data/mother_fqc_before/13.csv")
        # wideData_dir = pd.read_csv(filePath1)
        wideData_dir = wideData_dir.drop_duplicates(['upid'], keep='last').reset_index().drop(['index'], axis=1)
        wideData_dir[['upid']] = wideData_dir[['upid']].astype('str')
        wideData_dir = wideData_dir.set_index('upid')
        wideData = wideData_dir.dropna()
        # data = pd.read_csv(filePath2)
        data = pd.read_csv("/usr/src/data/mother_fqc_before/12.csv")
        data = data.drop_duplicates(['upid'], keep='last').reset_index().drop(['index'], axis=1)
        data[['upid']] = data[['upid']].astype('str')
        data[['upid']] = data[['upid']].astype('str')
        data = data.set_index('upid')
        result = {
            'data': data,
            'wideData': wideData
        }

        return result

