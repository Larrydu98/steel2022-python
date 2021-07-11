'''
SampleDivideController
'''
import numpy as np
from flask import json

from ..models import db
from ..models.sampleChooseModel import SampleChoose
from ..models.sampleDivideModel import SampleDivide
from ..models.modelModel import Model
from ..models.variableDataModel import VariableData
from ..models.quotaDataModel import QuotaData
from ..models.algorithmModel import Algorithm
from ..methods.sampleDivideMethods.train_test_split import TrainTestSplit


class SampleDivideController:
    '''
    SampleDivideController
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
        if modelName == 'train_test_split':
            model = TrainTestSplit(modelParams)
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
        sample_choose = SampleChoose.query.filter_by(model_id=row_id).first()
        sample_divide = SampleDivide.query.filter_by(model_id=row_id).first()
        model_detail = Model.query.filter_by(id=row_id).first()
        
        if sample_choose is None:
            result['msg'] = '此id在数据库中无对应的记录'
            return result
        else:
            variable_ids = json.loads(sample_choose.variable_ids)
            quota_ids = json.loads(sample_choose.quota_ids)
            date_range = sample_choose.date_range
        
        # 获取输入数据，并整理格式
        algorithm_input = {
            'X': [],
            'y': []
        }
        for variable_id in variable_ids:
            single_variable_infos =  VariableData.query.filter_by(variable_id=variable_id).order_by(VariableData.sample_time).all()
            single_variable_values = [item.value for item in single_variable_infos]
            algorithm_input['X'].append(single_variable_values)
        algorithm_input['X'] = np.array(algorithm_input['X']).T

        single_quota_infos =  QuotaData.query.filter_by(quota_id=quota_ids[0]).order_by(QuotaData.sample_time).all()
        algorithm_input['y'] = [item.actual_value for item in single_quota_infos]
        algorithm_input['y'] = np.array(algorithm_input['y']).T

        # 获取参数
        algorithm_name = Algorithm.query.get(sample_divide.algorithm_id).name
        algorithm_params = json.loads(sample_divide.algorithm_params)

        # 建模并调用
        instance = self.createModel(algorithm_name, algorithm_params)
        algorithm_output = instance.general_call(algorithm_input)

        # # 将数据及模型存到数据库中
        sample_divide.trains = json.dumps({
            'X_train': algorithm_output['X_train'].tolist(),
            'y_train': algorithm_output['y_train'].tolist()
        })
        sample_divide.tests = json.dumps({
            'X_test': algorithm_output['X_test'].tolist(),
            'y_test': algorithm_output['y_test'].tolist()
        })
        db.session.add(sample_divide)

        # 更新模型的step字段
        model_detail.step = 2
        db.session.add(model_detail)

        try:
            db.session.commit()
            result['status'] = True
        except:
            db.session.rollback()
            result['msg'] = '样本选择结果未存入进数据库'
        return result
