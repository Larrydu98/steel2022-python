'''
StrategySaveController
'''
import dill, pika, traceback, pickle
import numpy as np
from flask import json

from ..models import db
from ..models.algorithmModel import Algorithm
from ..models.algorithmChooseModel import AlgorithmChoose
from ..models.strategyModel import Strategy
from ..models.variableDataModel import VariableData
from ..models.quotaDataModel import QuotaData

from ..methods.sampleDivideMethods.train_test_split import TrainTestSplit
from ..methods.dataPreMethods.MinMaxzation import MinMaxzation
from ..methods.dataPreMethods.Normalization import Normalization
from ..methods.dataPreMethods.Standardization import Standardization
from ..methods.algorithmChooseMethods.GradientBoostingRegressor import GradientBoostingRegressor
from ..methods.algorithmChooseMethods.SVR import SVR


class StrategySaveController:
    '''
    StrategySaveController
    '''
    def __init__(self, row_id):
        self.strategy_info = Strategy.query.filter_by(id=row_id).first()
        self.diagram_model = json.loads(self.strategy_info.diagram_model)

        self.algorithm_by_id = {}
        algorithms = Algorithm.query.all()
        for item in algorithms:
            self.algorithm_by_id[item.id] = item

        self.all_node = {}
        self.node_by_order = {}
        self.input_node_keys = []
        self.output_node_keys = []

        # 在nodeDataArray中迭代，对模块进行处理，方便后续使用
        for item in self.diagram_model['nodeDataArray']:
            self.all_node[item['key']] = item

            # 为策略的dataSource模块新增value属性用来保存数据
            if item['type'] == 'dataSource':
                variable_ids = item['variableSelection']
                quota_ids = item['quotaSelection']
                date_range = item['dateRange']
                item['value'] = []

                for variable_id in variable_ids:
                    single_variable_infos =  VariableData.query.filter_by(variable_id=variable_id).order_by(VariableData.sample_time).all()
                    single_variable_values = [variable_info.value for variable_info in single_variable_infos]
                    item['value'].append(single_variable_values)
                for quota_id in quota_ids:
                    single_quota_infos =  QuotaData.query.filter_by(quota_id=quota_id).order_by(QuotaData.sample_time).all()
                    single_quota_values = [quota_info.actual_value for quota_info in single_quota_infos]
                    item['value'].append(single_quota_values)
                item['value'] = np.array(item['value']).T

            if item['type'] == 'input':
                self.input_node_keys.append(item['key'])

            if item['type'] == 'output':
                self.output_node_keys.append(item['key'])
            
            if item['type'] == 'algorithm':
                self.node_by_order[item['order']] = item
            
            if item['type'] == 'model':
                self.node_by_order[item['order']] = item

                item['run'] = pickle.loads(AlgorithmChoose.query.filter_by(model_id=item['id']).first().model)
            
            if item['type'] == 'strategy':
                self.node_by_order[item['order']] = item

                item['run'] = dill.loads(Strategy.query.filter_by(id=item['id']).first().model)

    def run_algorithm(self, node):
        node_input = {}
        node_params = {}

        for input_name in list(node['input_desc'].keys()):
            source_key = node['input_desc'][input_name]['sourceKey']
            source_attr = node['input_desc'][input_name]['sourceAttr']

            if (source_attr):
                node_input[input_name] = self.all_node[source_key]['output'][source_attr]
            else:
                node_input[input_name] = self.all_node[source_key]['value']
        
        for params_name in list(node['params_desc'].keys()):
            node_params[params_name] = node['params_desc'][params_name]['defaultValue']

            source_key = node['params_desc'][params_name]['sourceKey']
            source_attr = node['params_desc'][params_name]['sourceAttr']

            if (source_key):
                if (source_attr):
                    node_params[params_name] = self.all_node[source_key]['output'][source_attr]
                else:
                    node_params[params_name] = self.all_node[source_key]['value']

        algorithm_name = self.algorithm_by_id[node['id']].name
        if algorithm_name == 'train_test_split':
            result = TrainTestSplit(node_params).general_call(node_input)
        elif algorithm_name == 'Standardization':
            result = Standardization().general_call(node_input)
        elif algorithm_name == 'Normalization':
            result = Normalization().general_call(node_input)
        elif algorithm_name == 'MinMaxzation':
            result = MinMaxzation().general_call(node_input)
        elif algorithm_name == 'GradientBoostingRegressor':
            result = GradientBoostingRegressor(node_params).general_call(node_input)
        elif algorithm_name == 'Support Vector Machines':
            result = SVR(node_params).general_call(node_input)
        
        return result

    def prepare_for_model(self, node):
        node_input = []

        for item in node['input_desc']:
            source_key = item['sourceKey']
            source_attr = item['sourceAttr']

            # 模型的输入是由X和y的值按照一定顺序组成的数组构成的
            if (source_attr):
                node_input.append(self.all_node[source_key]['output'][source_attr])
            else:
                node_input.append(self.all_node[source_key]['value'])
        
        return np.array([node_input])
    
    def prepare_for_strategy(self, node):
        node_input = {}

        for input_name in list(node['input_desc'].keys()):
            source_key = node['input_desc'][input_name]['sourceKey']
            source_attr = node['input_desc'][input_name]['sourceAttr']

            # 如果source_attr不是空字符串，说明数据的来源模块是algorithm、model或strategy
            if (source_attr):
                node_input[input_name] = self.all_node[source_key]['output'][source_attr]
            else:
                node_input[input_name] = self.all_node[source_key]['value']
        
        return node_input

    def execute_node(self, node):
        if node['type'] == 'algorithm':
            node['output'] = self.run_algorithm(node)

        elif node['type'] == 'model':
            node['output'] = node['run'](self.prepare_for_model(node))

        elif node['type'] == 'strategy':
            node['output'] = node['run'](self.prepare_for_strategy(node))

    def run(self):
        def generate_model(actual_input):
            for input_key in self.input_node_keys:
                self.all_node[input_key]['value'] = actual_input[self.all_node[input_key]['text']]

            for index in range(0, len(list(self.node_by_order.keys()))):
                self.execute_node(self.node_by_order[index])
            
            actual_output = {}
            for output_key in self.output_node_keys:
                source_key = self.all_node[output_key]['sourceKey']
                source_attr = self.all_node[output_key]['sourceAttr']

                if (source_attr):
                    if self.all_node[source_key]['type'] == 'model':
                        actual_output[self.all_node[output_key]['text']] = self.all_node[source_key]['output'][int(source_attr)]
                    else:
                        actual_output[self.all_node[output_key]['text']] = self.all_node[source_key]['output'][source_attr]
                else:
                    actual_output[self.all_node[output_key]['text']] = self.all_node[source_key]['value']

            return actual_output
        
        self.strategy_info.model = dill.dumps(generate_model)
        db.session.add(self.strategy_info)

        try:
            db.session.commit()
        except:
            print(traceback.format_exc())
            db.session.rollback()



    
