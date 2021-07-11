'''
SampleChooseController
'''
import time
import copy
import traceback
import pickle
import numpy as np
import uuid
from flask import json

# from ..models import db
from ..models.dataPreModel import DataPre
from ..models.algorithmChooseModel import AlgorithmChoose
from ..models.modelModel import Model
from ..models.modelEvaluateModel import ModelEvaluate
from ..models.algorithmModel import Algorithm
from ..methods.algorithmChooseMethods.GradientBoostingRegressor import GradientBoostingRegressor
from ..methods.algorithmChooseMethods.SVR import SVR




class AlgorithmChooseController:
    '''
    AlgorithmChooseController
    '''
    def __init__(self, row_id):
        self.idIllegal = False
        
        if isinstance(row_id, str):
            self.row_id = row_id 
        else:
            self.idIllegal = True

    # 创建模型实例   
    def createModel(self, modelName, modelParams):
        '''
        createModel
        '''
        if modelName == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(modelParams)
        elif modelName == 'Support Vector Machines':
            model = SVR(modelParams)
        return model 
          
    def run(self):
        '''
        run
        '''
        # 判断输入id是否正确
        result = {
            'status': False,
            'msg': ''
        }
        if self.idIllegal:
            result['msg'] = '输入id错误'
            return result
        
        # 根据id到数据库取得记录
        row_id = self.row_id
        algorithm_choose = AlgorithmChoose.query.filter_by(model_id=row_id).first()
        data_pre = DataPre.query.filter_by(model_id=row_id).first()
        model_detail = Model.query.filter_by(id=row_id).first()

        if algorithm_choose is None:
            result['msg'] = '此id在数据库中无对应的记录'
            return result
        else:
            algorithm_info = json.loads(algorithm_choose.algorithms)
            train_data = json.loads(data_pre.train_output)
            test_data = json.loads(data_pre.test_output)

        # 获取输入数据
        # (1)获取待预处理的训练集数据
        X_train = train_data['X_train_pre']
        y_train = train_data['y_train']
        # (2)获取待预处理的测试集数据
        X_test = test_data['X_test_pre']
        y_test = test_data['y_test']


        # 选择模型参数
        algorithm_id = list(algorithm_info.keys())[0]
        algorithm_name = Algorithm.query.get(algorithm_id).name
        algorithm_params = algorithm_info[algorithm_id]

        # 建模
        instance = self.createModel(algorithm_name, algorithm_params)
        algorithm_input = {
            'X_train': np.array(X_train),
            'y_train': np.array(y_train)
        }
        algorithm_output = instance.general_call(algorithm_input)
        model = copy.copy(algorithm_output['model'].predict)
        
        # 预测
        y_train_predict = algorithm_output['model'].predict(X_train)
        y_test_predict = algorithm_output['model'].predict(X_test)
        pre_result = {
            'y_train_predict': np.around(y_train_predict, decimals=2).tolist(),
            'y_train':np.array(y_train).astype('float64').tolist(),
            'y_test_predict': np.around(y_test_predict, decimals=2).tolist(),
            'y_test': np.array(y_test).astype('float64').tolist()
        }
        # # 将数据及模型存到数据库中
        algorithm_choose.model = pickle.dumps(model)
        db.session.add(algorithm_choose)

        # 更新模型的step字段
        model_detail.step = 4
        db.session.add(model_detail)

        time_now = int(time.time())
        time_local = time.localtime(time_now)
        model_evaluate_record = ModelEvaluate.query.filter_by(model_id=row_id).first()
        if model_evaluate_record:
            model_evaluate_record.result = json.dumps(pre_result)
            model_evaluate_record.created_time = time_local
            model_evaluate_record.updated_time = time_local
            db.session.add(model_evaluate_record)
        else:
            id = uuid.uuid1()
            model_evaluate_info = ModelEvaluate(str(id), row_id, json.dumps(pre_result), time_local, time_local)
            db.session.add(model_evaluate_info)
        try:
            db.session.commit()
            result['status'] = True
        except:
            db.session.rollback()
            print(traceback.format_exc())
            result['msg'] = '样本选择结果未存入进数据库'
        return result
