import numpy as np
import pandas as pd
import datetime as dt
from ..models.plateYieldStaisticsDB import *
import math


class ComputePlateYieldStaisticsData:

    def __init__(self, time_diff, start_time, end_time):
        self.plate_yield_data = getPlateYieldStaisticsData.getPlateYieldStaisticsData(start_time, end_time)
        self.time_diff = int(time_diff)
        self.start_time = start_time
        self.end_time = end_time

    def getPlateYieldData(self):
        month_data = {'toc': [], 'upid': [], 'flag': []}
        for item in self.plate_yield_data[0]:
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
        month_data = pd.DataFrame(month_data)
        good_flag, bad_flag, no_flag, endTimeOutput = self.run(self.time_diff, month_data)
        return {'endTimeOutput': endTimeOutput, 'good_flag': good_flag, 'bad_flag': bad_flag,
                'no_flag': no_flag}, 200, {'Access-Control-Allow-Origin': '*'}

    def run(self, timeDiff, upid_CR_FQC):
        '''
        run
        '''
        startTime = dt.datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')
        endTime = dt.datetime.strptime(self.end_time, '%Y-%m-%d %H:%M:%S')

        hours = math.ceil((endTime - startTime).total_seconds() // 3600 / timeDiff)
        upid_CR_FQC['toc'] = pd.to_datetime(upid_CR_FQC['toc'], format='%Y-%m-%d %H:%M:%S')

        startPostion = startTime
        if hours == 1:
            endPostion = endTime
        else:
            endPostion = startTime + dt.timedelta(hours=timeDiff)

        good_flag = []
        bad_flag = []
        no_flag = []
        endTimeOutput = []

        for i in range(hours):
            good_flag.append(len(upid_CR_FQC[(upid_CR_FQC['toc'] > startPostion) & (upid_CR_FQC['toc'] < endPostion) & (
                    upid_CR_FQC['flag'] == 1)]))
            bad_flag.append(len(upid_CR_FQC[(upid_CR_FQC['toc'] > startPostion) & (upid_CR_FQC['toc'] < endPostion) & (
                    upid_CR_FQC['flag'] == 0)]))
            no_flag.append(len(upid_CR_FQC[(upid_CR_FQC['toc'] > startPostion) & (upid_CR_FQC['toc'] < endPostion) & (
                    upid_CR_FQC['flag'] == 404)]))
            endTimeOutput.append(str(endPostion))

            startPostion = endPostion
            if i == hours - 1:
                endPostion = endTime
            else:
                endPostion = startPostion + dt.timedelta(hours=timeDiff)

        return good_flag, bad_flag, no_flag, endTimeOutput


class getDataPlateYieldAndFlag:
    '''
    getDataPlateYieldAndFlag
    '''

    def __init__(self, startTime, endTime):
        self.startTime = startTime
        self.endTime = endTime

    def run(self, timeDiff, upid_CR_FQC):
        '''
        run
        '''
        startTime = dt.datetime.strptime(self.startTime, '%Y-%m-%d %H:%M:%S')
        endTime = dt.datetime.strptime(self.endTime, '%Y-%m-%d %H:%M:%S')

        hours = math.ceil((endTime - startTime).total_seconds() // 3600 / timeDiff)
        upid_CR_FQC['toc'] = pd.to_datetime(upid_CR_FQC['toc'], format='%Y-%m-%d %H:%M:%S')

        startPostion = startTime
        if hours == 1:
            endPostion = endTime
        else:
            endPostion = startTime + dt.timedelta(hours=timeDiff)

        good_flag = []
        bad_flag = []
        no_flag = []
        endTimeOutput = []

        for i in range(hours):
            good_flag.append(len(upid_CR_FQC[(upid_CR_FQC['toc'] > startPostion) & (upid_CR_FQC['toc'] < endPostion) & (
                    upid_CR_FQC['flag'] == 1)]))
            bad_flag.append(len(upid_CR_FQC[(upid_CR_FQC['toc'] > startPostion) & (upid_CR_FQC['toc'] < endPostion) & (
                    upid_CR_FQC['flag'] == 0)]))
            no_flag.append(len(upid_CR_FQC[(upid_CR_FQC['toc'] > startPostion) & (upid_CR_FQC['toc'] < endPostion) & (
                    upid_CR_FQC['flag'] == 404)]))
            endTimeOutput.append(str(endPostion))

            startPostion = endPostion
            if i == hours - 1:
                endPostion = endTime
            else:
                endPostion = startPostion + dt.timedelta(hours=timeDiff)

        return good_flag, bad_flag, no_flag, endTimeOutput
