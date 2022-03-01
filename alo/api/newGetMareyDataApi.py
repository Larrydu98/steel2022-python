'''
sampleChoose
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.newGetMareyDataController import newComputeMareyData
import pika, traceback

parser = reqparse.RequestParser(trim=True, bundle_errors=True)


# # 根目录
# @app.route('/')


class newGetMareyStationsDataApi(Resource):
    '''
    AlgorithmChoose
    '''

    def post(self, upid, start_time, end_time):  # , steelspec, tgtplatethickness
        parser.add_argument("steelspec", type=str, required=True)
        parser.add_argument("tgtplatethickness", type=str, required=True)
        # parser.add_argument("status_cooling", type=str, required=True)
        args = parser.parse_args(strict=True)
        steelspec = args["steelspec"]
        tgtplatethickness = json.loads(args["tgtplatethickness"])

        res = newComputeMareyData(type="stations",
                                  upid=upid,
                                  start_time=start_time,
                                  end_time=end_time,
                                  steelspec=steelspec,
                                  tgtplatethickness=tgtplatethickness
                                  )
        # res.printData()
        status, stations_result = res.newGetMareyStations()
        # print(stations_result)

        return stations_result, status, {'Access-Control-Allow-Origin': '*'}


class newGetMareyTimesDataApi(Resource):
    '''
    AlgorithmChoose
    '''

    def post(self, upid, start_time, end_time, compressed_factor):
        parser.add_argument("steelspec", type=str, required=True)
        parser.add_argument("tgtplatethickness", type=str, required=True)
        # parser.add_argument("status_cooling", type=str, required=True)
        args = parser.parse_args(strict=True)
        steelspec = args["steelspec"]
        tgtplatethickness = json.loads(args["tgtplatethickness"])

        res = newComputeMareyData(type="times",
                                  upid=upid,
                                  start_time=start_time,
                                  end_time=end_time,
                                  steelspec=steelspec,
                                  tgtplatethickness=tgtplatethickness
                                  )

        status, stations_result = res.newGetMareyTimes(compressed_factor)

        return stations_result, status, {'Access-Control-Allow-Origin': '*'}


api.add_resource(newGetMareyStationsDataApi, '/v1.0/newGetMareyStationsDataApi/<upid>/<start_time>/<end_time>/')
api.add_resource(newGetMareyTimesDataApi,
                 '/v1.0/newGetMareyTimesDataApi/<upid>/<start_time>/<end_time>/<compressed_factor>')
