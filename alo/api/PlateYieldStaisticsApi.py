from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.PlateYieldStaisticsController import ComputePlateYieldStaisticsData

parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class PlateYieldStaisticsApi(Resource):
    def get(self, time_diff, start_time, end_time):
        res = ComputePlateYieldStaisticsData(time_diff, start_time, end_time)
        result = res.getPlateYieldData()

        return result, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(PlateYieldStaisticsApi, '/v2.0/PlateYieldStaisticsApi/<time_diff>/<start_time>/<end_time>/')
