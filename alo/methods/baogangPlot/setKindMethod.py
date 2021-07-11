'''
setKindMethod
'''
# https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
import pandas as pd
import numpy as np

class setKindMethod:
    '''
    setKindMethod
    '''

    def __init__(self):
        self.paramsIllegal = False
        try:
            # 如果以后加入参数，在这里加入
            print('setKind方法暂时没有参数')
        except Exception:
            self.paramsIllegal = True

    def general_call(self, custom_input):
        '''
        general_call
        '''
        data = custom_input['data']
        # self.model = MinMaxScaler().fit(np.array(custom_input['X']).astype(np.float64))
        # array = self.model.transform(np.array(custom_input['X']).astype(np.float64))
        targetSteel = data.loc[custom_input['upid']]['steelspec']
        targetThick = data.loc[custom_input['upid']]['tgtplatethickness2']
        targetWidth = data.loc[custom_input['upid']]['tgtwidth']
        thickRule = np.array([0.0099+item*0.007 for item in range(6)])
        widthRule = np.array([1.68+item*0.62 for item in range(6)])
        targetThickList = np.array([targetThick]*len(thickRule))
        targetWidthList = np.array([targetWidth]*len(widthRule))
        try:
            thickIndex = list(np.where((targetThickList - thickRule)<0,1,-1)).index(1)
        except:
            thickIndex = len(thickRule)
        try:
            widthIndex = list(np.where((targetWidthList - widthRule)<0,1,-1)).index(1)
        except:
            widthIndex = len(widthIndex)
        if thickIndex==0:
            highThickRule = thickRule[0]
            lowThickRule = 0
        elif thickIndex==len(thickRule):            
            highThickRule = thickRule[0]
            lowThickRule = 0
        else: 
            highThickRule = thickRule[thickIndex]
            lowThickRule = thickRule[thickIndex-1]
        if widthIndex==0:
            highWidthRule = widthRule[0]
            lowWidthRule = 0
        elif thickIndex==len(thickRule):            
            highThickRule = thickRule[0]
            lowThickRule = 0
        else: 
            highWidthRule = widthRule[widthIndex]
            lowWidthRule = widthRule[widthIndex-1]
        return {
            'targetSteel': targetSteel,
            'highThickRule': highThickRule,
            'lowThickRule': lowThickRule,
            'highWidthRule': highWidthRule,
            'lowWidthRule': lowWidthRule
        }
