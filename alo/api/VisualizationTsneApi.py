from flask_restful import Resource, reqparse
from . import api
from ..controller.VisualizationTsneController import VisualizationTsneController
from ..utils import postArgs
parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class VisualizationTsneApi(Resource):
    def post(self, start_time, end_time,minute_diff, merge_limit, merge_conflict):
        new_args, cooling = postArgs(parser)
        res = VisualizationTsneController(start_time, end_time,minute_diff, merge_limit, merge_conflict, new_args, cooling)
        result = res.visualizationTsenController()
        if len(result) <= 5:
            return {}, 202, {'Access-Control-Allow-Origin': '*'}

        return result, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(VisualizationTsneApi, '/v2.0/VisualizationTsneApi/<start_time>/<end_time>/<minute_diff>/<merge_limit>/<merge_conflict>/')
