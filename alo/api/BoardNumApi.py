'''
dataPre
'''
from flask_restful import Resource, reqparse
from . import api
import pika
import traceback
from ..controller.BoardNumController import ComputeBoardNum

parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class BoardNumApi(Resource):

    def post(self, upid, limit):

        computeBoardNum_instance = ComputeBoardNum(upid)

        selection = ['dd.upid', 'dd.status_fqc', 'dd.fqc_label']
        data = computeBoardNum_instance.getData(parser, selection, limit)

        try:
            status_code, result = computeBoardNum_instance.getGoodNum(data)
        except Exception:
            print(traceback.format_exc())
        # if (result == 400):
        #     return "", 204, {'Access-Control-Allow-Origin': '*'}
        # if (len(good_num) < 5):
        #     return {'result': result, 'outOfGau': outOfGau, 'PCAT2': PCAT2, 'PCASPE': PCASPE}, 202, {'Access-Control-Allow-Origin': '*'}
        return result, status_code, {'Access-Control-Allow-Origin': '*'}


api.add_resource(BoardNumApi, '/v1.0/baogangPlot/boardNumApi/<upid>/<limit>')
