from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.diagnosisDataController import diagnosisDataComputer

parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class diagnosisDataByUpidApi(Resource):
    '''
    getFlag
    '''

    def post(self, sort_type, plate_limit):
        res = diagnosisDataComputer(parser, sort_type, plate_limit)
        diag_result, status_code,  = res.getdiagnosisData()
        # data = res.
        # res = ComputeThicknessData(parser, plate_limit, day_limit, hour_limit)
        # data = res.printData
        return diag_result, status_code, {'Access-Control-Allow-Origin': '*'}


api.add_resource(diagnosisDataByUpidApi, '/v2.0/DiagnosisDataApi/<sort_type>/<plate_limit>/')
