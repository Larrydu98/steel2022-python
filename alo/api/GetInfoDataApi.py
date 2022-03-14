from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.GetInfoDataController import GetInfoDataController
import datetime as dt
parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class GetInfoDataApi(Resource):
    def post(self, start_time, end_time, merge_limit, merge_conflict):
        label = ["tgtthickness", "tgtwidth", 'tgtlength', "platetype", "cooling"]
        for index in label:
            parser.add_argument(index, type=str, required=True)
        args = parser.parse_args(strict=True)
        for arg_index in args:
            if arg_index == 'platetype':
                continue
            if arg_index == 'cooling':
                args['cooling'] = int(args['cooling'])
                continue
            args[arg_index] = eval(args[arg_index])
        end_time = start_time
        start_time = dt.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        start_time = str(start_time.replace(month=start_time.month - 1))

        res = GetInfoDataController(args=args,
                                    start_time=start_time,
                                    end_time=end_time,
                                    merge_limit=merge_limit,
                                    merge_conflict=merge_conflict
                                    )
        data = res.getInfoData()
        return data, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(GetInfoDataApi, '/v2.0/InfoDataApi/<start_time>/<end_time>/<merge_limit>/<merge_conflict>/')
