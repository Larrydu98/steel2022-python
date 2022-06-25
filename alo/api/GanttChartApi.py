from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.newGanttChartDataControler import getGanttChartData
from ..utils import postArgs

parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class GanttChartApi(Resource):
    def post(self, start_time, end_time,minute_diff, merge_limit, merge_conflict):
        new_args, cooling = postArgs(parser)
        res = getGanttChartData(start_time, end_time,minute_diff, merge_limit, merge_conflict, new_args, cooling)
        result = res.getGanttChartData()

        # result = 'ok'
        return result, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(GanttChartApi, '/v2.0/GanttChartApi/<start_time>/<end_time>/<minute_diff>/<merge_limit>/<merge_conflict>/')
