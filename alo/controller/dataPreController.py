'''
DataPreController
'''
import numpy as np
from flask import json

# from ..models import db
from ..models.dataPreModel import DataPre
from ..models.sampleDivideModel import SampleDivide
from ..models.modelModel import Model
from ..models.algorithmModel import Algorithm
from ..methods.dataPreMethods.MinMaxzation import MinMaxzation
from ..methods.dataPreMethods.Normalization import Normalization
from ..methods.dataPreMethods.Standardization import Standardization




class DataPreController:
    '''
    DataPreController
    '''
    def __init__(self, row_id):
        self.idIllegal = False
        
        if isinstance(row_id, str):
            self.row_id = row_id 
        else:
            self.idIllegal = True

    # 创建模型实例   
    def createModel(self, modelName):
        '''
        createModel
        '''
        if modelName == 'Standardization':
            model = Standardization()
        elif modelName == 'Normalization':
            model = Normalization()
        elif modelName == 'MinMaxzation':
            model = MinMaxzation()
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
        data_pre = DataPre.query.filter_by(model_id=row_id).first()
        sample_divide = SampleDivide.query.filter_by(model_id=row_id).first()
        model_detail = Model.query.filter_by(id=row_id).first()

        if sample_divide is None:
            result['msg'] = '此id在数据库中无对应的记录'
            return result
        else:
            algorithm_info = json.loads(data_pre.algorithms)
            pre_data_train = json.loads(sample_divide.trains)
            pre_data_test = json.loads(sample_divide.tests)

        # 获取输入数据
        # (1)获取待预处理的训练集数据
        X_train = pre_data_train['X_train']
        y_train = pre_data_train['y_train']
        # (2)获取待预处理的测试集数据
        X_test = pre_data_test['X_test']
        y_test = pre_data_test['y_test']
        # 选择模型参数
        algorithm_name = Algorithm.query.get(list(algorithm_info.keys())[0]).name

        # 建模
        instance = self.createModel(algorithm_name)
        X_train_pre = instance.general_call({ 'X': X_train })['array']
        X_test_pre = instance.general_call({ 'X': X_test })['array']

        # # 将数据及模型存到数据库中
        data_pre.train_output = json.dumps({
            'X_train_pre': X_train_pre.tolist(),
            'y_train': y_train
        })
        data_pre.test_output = json.dumps({
            'X_test_pre': X_test_pre.tolist(),
            'y_test': y_test
        })
        db.session.add(data_pre)

        # 更新模型的step字段
        model_detail.step = 3
        db.session.add(model_detail)

        try:
            db.session.commit()
            result['status'] = True
        except:
            db.session.rollback()
            result['msg'] = '样本选择结果未存入进数据库'
        return result
