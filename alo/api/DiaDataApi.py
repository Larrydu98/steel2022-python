from ..controller.DiaDataController import DiaDataController
from ..utils import postArgs
from flask_restful import Resource, reqparse
from flask import json
from . import api


parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class DiaDataApi(Resource):
    def post(self, start_time, end_time, minute_diff, merge_limit, merge_conflict, plate_limit):
        new_args, cooling = postArgs(parser)
        res_class = DiaDataController(start_time, end_time, minute_diff, merge_limit, merge_conflict, plate_limit, new_args, cooling)
        res = res_class.getDiaData()

        return res


api.add_resource(DiaDataApi,
                 '/v2.0/DiaDataApi/<start_time>/<end_time>/<minute_diff>/<merge_limit>/<merge_conflict>/<plate_limit>/')
