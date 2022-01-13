'''
sampleChoose
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.getMareyDataController import ComputeMareyData
import pika, traceback
from ..controller.iconChangeController import eventChangeDataController
import json
import numpy as np

# parser = reqparse.RequestParser(trim=True, bundle_errors=True)
#

# # 根目录
# @app.route('/')
# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(NpEncoder, self).default(obj)
class eventDataApi(Resource):
    '''
    获取事件数据的api
    '''
    def get(self, start_time, end_time, compressed_factor, cooling_constrain, zeropoint_constrain):



        res = eventChangeDataController(type="times",
                                       start_time=start_time,
                                       end_time=end_time)
        result = res.newGetMareyTimes(compressed_factor, cooling_constrain, zeropoint_constrain)
        # result = json.dumps(result, cls=NpEncoder)
        # print(result)
        return result, 200, {'Access-Control-Allow-Origin': '*'}




api.add_resource(eventDataApi, '/v1.0/eventDataApi/<start_time>/<end_time>/<compressed_factor>/<cooling_constrain>/<zeropoint_constrain>/')
