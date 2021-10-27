'''
dataPre
'''
import os
import pika
import traceback
import pandas as pd
import numpy as np
from flask_restful import Resource, reqparse

from . import api
from ..controller.baogangPlot.createMonitorResuByTime import createMonitorResu
from ..models.monitorResuModel import parserMoniRequArge


parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class monitorDataByTimeApi(Resource):
    '''
    diagnosisDataApi
    '''
    def post(self, start_time, end_time, sorttype, limit):
        """ 
        post
        ---
        tags:
            - 诊断视图数据
        parameters:
            - in: body
              name: body
              schema:
              properties:
                  upid:
                #   type: string
                #   default: 1
                #   description: 数据预处理
                #   task_id:
                #   type: string
                #   default: 1
                #   description: 任务id
              required: true
        responses:
          200:
            description: 执行成功
        """

        request_bodys, tocs = parserMoniRequArge(parser)

        # data, _, status_cooling, fqcflag = new_modeldata(parser,
        #                                         ['dd.upid', 'lmpd.productcategory', 'dd.tgtwidth','dd.tgtlength',
        #                                          'dd.tgtthickness','dd.stats','dd.fqc_label', 'dd.toc'],
        #                                         limit)
        #
        # if (len(data) == 0):
        #     return {}, 204, {'Access-Control-Allow-Origin': '*'}
        #
        try:
        #     _data_names = []
        #     if status_cooling == 0:
        #         _data_names = data_names
        #     elif status_cooling == 1:
        #         _data_names = without_cooling_data_names

            createMonitorResu_instance = createMonitorResu(start_time, end_time)
            monitor_result, status_code = createMonitorResu_instance.run(request_bodys, tocs, sorttype, limit)

            return monitor_result, status_code, {'Access-Control-Allow-Origin': '*'}

        except Exception:
            print(traceback.format_exc())
            return {}, 500, {'Access-Control-Allow-Origin': '*'}


api.add_resource(monitorDataByTimeApi, '/v1.0/baogangPlot/monitordatabytime/<start_time>/<end_time>/<sorttype>/<limit>')
