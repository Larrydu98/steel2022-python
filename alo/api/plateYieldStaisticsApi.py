'''
plateYieldStaisticsApi
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
# from ..controller.modelTransferCsvController import modelTransferCsvController
# from ..controller.getPlateYieldStaisticsAndFlagController import getDataPlateAndFlagCRAndFqc
from ..controller.getPlateYieldStaisticsAndFlagController import getDataPlateYieldAndFlag
from ..utils import getData, new_getData
from ..utils import getFlagArr
from ..utils import ref
import pandas as pd
import numpy as np
parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class plateYieldStaistics(Resource):
    '''
    plateYieldStaistics
    '''
    def get(self,timeDiff,startTime,endTime):
        """
        get
        ---
        tags:
            - 时间线图
        parameters:
            - in: path
              name: timeDiff
              required: true
              description: 时间间隔
              type: string
            - in: path
              name: startTime
              required: true
              description: 开始时间
              type: string
            - in: path
              name: endTime
              required: true
              description: 结束时间
              type: string
        responses:
            200:
                description: 执行成功
        """
        # return {'hello': 'world'}

        timeDiff = int(timeDiff)

        # DataPlateAndFlagCRAndFqc = getDataPlateAndFlagCRAndFqc(startTime,endTime)
        # month_data = DataPlateAndFlagCRAndFqc.run()




        # month_data is the data read from database
        tocSelect = [startTime, endTime]
        # ismissing={'all_processes_statistics_ismissing':True,'cool_ismissing':True,'fu_temperature_ismissing':True,'m_ismissing':True,'fqc_ismissing':True}
        ismissing = {}
        data = new_getData(['upid', 'toc', 'status_fqc', 'fqc_label'], ismissing, [], [], [], tocSelect, [], [], '', '')
        month_data = {
          'toc': [],
          'upid': [],
          'flag': []
        }
        for item in data[0]:
            # print(item[0])
            month_data['upid'].append(item[0])
            month_data['toc'].append(item[1])
            if item[2] == 1:  # status_fqc
                month_data['flag'].append(404)
            elif item[2] == 0:
                if np.array(item[3]['method1']['data']).sum() == 5:  # fqc_label
                    month_data['flag'].append(1)
                else:
                    month_data['flag'].append(0)

          # msum = 0
          # month_data['toc'].append(item[1])
          # month_data['upid'].append(item[0])
          # flagArr = getFlagArr(item[2]['method1'])
          # for label in flagArr:
          #   msum = msum + label
          # if (msum >= ref):
          #   month_data['flag'].append(1)
          # else:
          #   month_data['flag'].append(0)
        month_data = pd.DataFrame(month_data)
        # print(month_data['flag'].shape)
        PlateYieldStaistics = getDataPlateYieldAndFlag(startTime,endTime)
        good_flag,bad_flag,no_flag,endTimeOutput = PlateYieldStaistics.run(timeDiff, month_data)
        # good_flag,bad_flag,endTimeOutput = PlateYieldStaistics.run(timeDiff,month_data)


        return {'endTimeOutput': endTimeOutput,'good_flag': good_flag,'bad_flag': bad_flag,'no_flag':no_flag}, 200, {'Access-Control-Allow-Origin': '*'}



api.add_resource(plateYieldStaistics, '/v1.0/model/plateYieldStaistics/<timeDiff>/<startTime>/<endTime>/')
