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
    def run(self):
        '''
        run
        '''
        filePath_1 = "/usr/src/data/mother_fqc_before/10.json"
        # filePath_1 = r'/alo/data/model_raw_data_190523.json'
        # filePath_2 = r'D:\document\BAOGANGDATA\plate_type_distribution190530.csv'
        filePath_2 = "/usr/src/data/mother_fqc_before/11.csv"
        # filePath_2 = r'/alo/data/plate_type_distribution190530.csv'
        # filePath3 = r'D:\document\ALO_PYTHON\alo\api\wide_117_data_flag_20190831.csv'
        # filePath4 = r'D:\document\ALO_PYTHON\alo\api\wide_117_data_flag_20190831.csv'

        parser = reqparse.RequestParser(trim=True, bundle_errors=True)
        # targetData_dir = pd.read_csv(filePath_2)
        targetData_dir = pd.read_csv("/usr/src/data/mother_fqc_before/11.csv")
        targetData_dir = targetData_dir.drop_duplicates(['upid'], keep='last').reset_index().drop(['index'], axis=1)
        targetData_dir[['upid']] = targetData_dir[['upid']].astype('str')
        targetData_dir = targetData_dir.set_index('upid')
        targetData = targetData_dir.dropna()
        # data = pd.read_json(filePath_1)
        data = pd.read_json("/usr/src/data/mother_fqc_before/10.json")
        data = data.drop_duplicates(['upid'], keep='last').reset_index().drop(['index'], axis=1)
        data[['upid']] = data[['upid']].astype('str')
        data[['upid']] = data[['upid']].astype('str')
        data = data.set_index('upid')

        result = {
            'data': data,
            'targetData': targetData
        }
        return result

