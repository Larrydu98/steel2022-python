'''
detailProcess
'''
# https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
import pandas as pd
import numpy as np


class detailProcess:
    '''
    detailProcess
    '''

    def __init__(self):
        self.paramsIllegal = False
        try:
            # 如果以后加入参数，在这里加入
            print('detailProcess方法暂时没有参数')
        except Exception:
            self.paramsIllegal = True

    def general_call(self, custom_input):
        '''
        general_call
        '''
        data = custom_input['data']                     #dataFrame格式
        steelKindUpid = custom_input['kindUpid']        #list
        targetData = custom_input['targetData']
        feature = custom_input['feature']
        upid = custom_input['upid']
        steelKindData = targetData.loc[steelKindUpid]
        feature_real = feature.split('-')[0] 
        ismissing = {'all_processes_statistics_ismissing':True,'cool_ismissing':True,'fu_temperature_ismissing':True,'m_ismissing':True,'fqc_ismissing':True}
        data = getData(['upid', 'platetype', 'tgtwidth','tgtlength','tgtthickness','all_processes_statistics','fqc_label', 'toc'], ismissing, [], [], [], [], [self.upid], [], '', '')
#       
        # print(data)
        dataLabel = 0
        dataFlag = getFlagArr(data[0][6]['method1'])
        dataAmount=0
        for j in dataFlag:
            dataAmount+=j
        if(dataAmount>=3):
            dataLabel = 1
         
        time = data[0][7]
        delta = datetime.timedelta(days = 15)
        start = str(time - delta)
        end = str(time + delta)

        otherPlate = data[0][1]
        otherPlate = ''
        otherWidth = [str(data[0][2] - 1), str(data[0][2] + 1)]
        # print(type(otherWidth[0]))
        otherLen = [str(data[0][3] - 5), str(data[0][3] + 5)]
        otherLen = []
        otherThick = [str(data[0][4] - 5), str(data[0][4] + 5)]
        otherSelectTime = [start, end]
        otherData = getData(['upid', 'platetype', 'tgtwidth','tgtlength','tgtthickness','all_processes_statistics','fqc_label'], 
        ismissing, otherWidth, otherLen, otherThick, otherSelectTime, [], otherPlate, '', '')

        # print(otherData)
        
        labelArr = []
        badBoardData = []
        goodBoardData = []
        badBoardId = []
        goodBoardId = []
        
        for item in otherData:
            flagArr = getFlagArr(item[6]['method1'])
            label=0
            amount=0
            for j in flagArr:
                amount+=j
            if(amount>=3):
                label=1
            labelArr.append(label)
            if (label > 0):
                badBoardId.append(item[0])
                badBoardData.append(item[5]['data'])
            else:
                goodBoardId.append(item[0])
                goodBoardData.append(item[5]['data'])
        if (dataLabel == 1):
            badBoardId.append(data[0][0])
            badBoardData.append(data[0][5]['data'])
        else:
            goodBoardId.append(data[0][0])
            goodBoardData.append(data[0][5]['data'])
        badBoardData = np.array(badBoardData)
        goodBoardData = np.array(goodBoardData)
        allBoardId = goodBoardId + badBoardId

        
        # finalResult = []
            return {
                'goodData': goodData,
                'badData': badData,
                'testBoardData': testBoardData
            }
