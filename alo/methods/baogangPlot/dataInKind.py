'''
dataInKind
'''
# https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
import pandas as pd
import numpy as np

def findSameInList(lst, *lsts):
    iset = set(lst)
    for li in lsts:
        s = set(li)
        iset = iset.intersection(s)
    return list(iset)

class dataInKind:
    '''
    dataInKind
    '''

    def __init__(self):
        self.paramsIllegal = False
        try:
            # 如果以后加入参数，在这里加入
            print('dataInKind方法暂时没有参数')
        except Exception:
            self.paramsIllegal = True

    def general_call(self, custom_input):
        '''
        general_call
        '''
        data = custom_input['data']
        targetSteel = custom_input['targetSteel']
        highThickRule = custom_input['highThickRule']
        lowThickRule = custom_input['lowThickRule']
        highWidthRule = custom_input['highWidthRule']
        lowWidthRule = custom_input['lowWidthRule']

        # self.model = MinMaxScaler().fit(np.array(custom_input['X']).astype(np.float64))
        # array = self.model.transform(np.array(custom_input['X']).astype(np.float64))
        steelKindList = data['steelspec'].tolist()
        steelKind = list(set(steelKindList))
        steelList = data['steelspec'][data['steelspec']== targetSteel].index.tolist()
        lowList = data['tgtplatethickness2'][lowThickRule <= data['tgtplatethickness2']].index.tolist()
        upList = data['tgtplatethickness2'][data['tgtplatethickness2'] <= highThickRule].index.tolist()
        lowListW = data['tgtwidth'][lowWidthRule <= data['tgtwidth']].index.tolist()
        upListW = data['tgtwidth'][data['tgtwidth'] <= highWidthRule].index.tolist()
        finalResult = findSameInList(steelList,lowList, upList)
        # finalResult = []
        return {
            'finalResult': finalResult
        }
