'''
StrategyRunController
'''
import dill, pika, traceback, pickle
from flask import json

from ..models import db
from ..models.strategyModel import Strategy


class StrategyRunController:
    '''
    StrategyRunController
    '''
    def __init__(self, row_id):
        self.idIllegal = False
        
        if isinstance(row_id, str):
            self.row_id = row_id 
        else:
            self.idIllegal = True

    def run(self, input_obj):
        '''
        run
        '''
        result = {
            'status': False,
            'msg': ''
        }
        if self.idIllegal:
            result['msg'] = '输入id错误'
            return result

        strategy_info = Strategy.query.filter_by(id=self.row_id).first()
        strategy_model = pickle.loads(strategy_info.model)

        return strategy_model(input_obj)
