"""
Visualization
"""

from flask_restful import Resource, reqparse
from . import api
from ..controller.newVisualizationByBatchController import GetProcessVisualizationData

parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class RollVisualizationByBatch(Resource):
    def post(self, start_time, end_time, deviation, limitation):
        getProcessVisualizationData = GetProcessVisualizationData(parser, start_time, end_time, 'roll', deviation, limitation)
        status_code, data = getProcessVisualizationData.getRoll()

        return data, status_code, {'Access-Control-Allow-Origin': '*'}


class HeatVisualizationByBatch(Resource):
    def post(self, start_time, end_time, box_num, deviation, limitation):
        getProcessVisualizationData = GetProcessVisualizationData(parser, start_time, end_time, 'heat', deviation, limitation)
        status_code, data = getProcessVisualizationData.getHeat(int(box_num))

        return data, status_code, {'Access-Control-Allow-Origin': '*'}


class CoolVisualizationByBatch(Resource):
    def post(self, start_time, end_time, box_num, divide_percent, deviation, limitation):
        getProcessVisualizationData = GetProcessVisualizationData(parser, start_time, end_time, 'cool', deviation, limitation)
        status_code, data = getProcessVisualizationData.getCool(int(box_num), int(divide_percent))

        return data, status_code, {'Access-Control-Allow-Origin': '*'}


api.add_resource(RollVisualizationByBatch, '/v1.0/model/newRollVisualizationByBatch/<start_time>/<end_time>/<deviation>/<limitation>')
api.add_resource(HeatVisualizationByBatch, '/v1.0/model/newHeatVisualizationByBatch/<start_time>/<end_time>/<box_num>/<deviation>/<limitation>')
api.add_resource(CoolVisualizationByBatch, '/v1.0/model/newCoolVisualizationByBatch/<start_time>/<end_time>/<box_num>/<divide_percent>/<deviation>/<limitation>')
