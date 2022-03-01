from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.GanttChartDataControler import getGanttChartData

parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class GanttChartApi(Resource):
    def get(self,start_time, end_time):
        res = getGanttChartData(start_time,end_time)
        result = res.getGanttChartData()
        result = res.getGanttChartData()

        # result = 'ok'
        return result, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(GanttChartApi, '/v2.0/GanttChartApi/<start_time>/<end_time>/')
