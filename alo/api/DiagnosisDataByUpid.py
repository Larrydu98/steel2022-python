from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.newdiagnosisDataController import newdiagnosisDataComputer

parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class diagnosisDataByUpidApi(Resource):
    '''
    getFlag
    '''

    def post(self, start_time, end_time, merge_limit, merge_conflict, plate_limit):
        res = newdiagnosisDataComputer(parser, start_time, end_time, merge_limit, merge_conflict, plate_limit)
        # diag_result, status_code, = res.getdiagnosisData()
        # data = res.
        # res = ComputeThicknessData(parser, plate_limit, day_limit, hour_limit)
        # data = res.printData
        # return diag_result, status_code, {'Access-Control-Allow-Origin': '*'}
        data = res.diagnosisDataController()
        return data, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(diagnosisDataByUpidApi,
                 '/v2.0/DiagnosisDataApi/<start_time>/<end_time>/<merge_limit>/<merge_conflict>/<plate_limit>/')
