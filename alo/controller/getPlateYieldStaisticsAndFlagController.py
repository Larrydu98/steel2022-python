'''
modelTransferController
'''
import pandas as pd
import datetime
import math


class getDataPlateYieldAndFlag:
    '''
    getDataPlateYieldAndFlag
    '''

    def __init__(self, startTime, endTime):
        self.startTime = startTime
        self.endTime = endTime

    def run(self,timeDiff,upid_CR_FQC):
        '''
        run
        '''
        startTime = datetime.datetime.strptime(self.startTime, '%Y-%m-%d %H:%M:%S')
        endTime = datetime.datetime.strptime(self.endTime, '%Y-%m-%d %H:%M:%S')

        hours = math.ceil((endTime - startTime).total_seconds() // 3600 / timeDiff)
        upid_CR_FQC['toc'] = pd.to_datetime(upid_CR_FQC['toc'], format='%Y-%m-%d %H:%M:%S')

        startPostion = startTime
        if hours == 1:
            endPostion = endTime
        else:
            endPostion = startTime + datetime.timedelta(hours=timeDiff)

        good_flag = []
        bad_flag  = []
        no_flag   = []
        endTimeOutput = []

        for i in range(hours):
            good_flag.append(len(upid_CR_FQC[(upid_CR_FQC['toc'] > startPostion) & (upid_CR_FQC['toc'] < endPostion) & (upid_CR_FQC['flag'] == 1)]))
            bad_flag.append(len(upid_CR_FQC[(upid_CR_FQC['toc'] > startPostion) & (upid_CR_FQC['toc'] < endPostion) & (upid_CR_FQC['flag'] == 0)]))
            no_flag.append(len(upid_CR_FQC[(upid_CR_FQC['toc'] > startPostion) & (upid_CR_FQC['toc'] < endPostion) & (upid_CR_FQC['flag'] == 404)]))
            endTimeOutput.append(str(endPostion))


            startPostion = endPostion
            if i == hours - 1:
                endPostion = endTime
            else:
                endPostion = startPostion + datetime.timedelta(hours=timeDiff)


        return good_flag,bad_flag,no_flag,endTimeOutput

# instance = modelTransferController('2018-09-01 00:00:00', '2018-09-02 00:00:00')
# instance.run()