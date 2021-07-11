'''
prepProData
'''
# https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
import pandas as pd
import numpy as np


class prepProData:
    '''
    prepProData
    '''

    def __init__(self):
        self.paramsIllegal = False
        try:
            # 如果以后加入参数，在这里加入
            print('prepProData方法暂时没有参数')
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
        if len(np.array(data.loc[upid][feature_real]).shape) == 3:
            feature_num = int(feature.split('-')[1])
            data_upid_all_good = []
            data_upid_all_bad = []
            goodDataUpid = steelKindData[steelKindData['upid_flag']==1].index
            goodData = data.loc[goodDataUpid][feature_real]
            badDataUpid = steelKindData[steelKindData['upid_flag']==0].index
            badData = data.loc[badDataUpid][feature_real]
            testBoardData = data.loc[upid][feature_real][feature_num]
            for i in range(len(goodData)): 
                data_upid_one_good = {'upid':goodData.index[i],
                                    'data':goodData[i][feature_num]}
                data_upid_all_good.append(data_upid_one_good)
            for i in range(len(badData)): 
                data_upid_one_bad = {'upid':badData.index[i],
                                    'data':badData[i][feature_num]}
                data_upid_all_bad.append(data_upid_one_bad)
            return {
                'goodData': pd.DataFrame(data_upid_all_good).set_index('upid')['data'],
                'badData': pd.DataFrame(data_upid_all_bad).set_index('upid')['data'],
                'testBoardData': testBoardData
                # 'goodData': data_upid_all_good,
                # 'badData': data_upid_all_bad,
                # 'testBoardData': testBoardData
            }
        else:
            goodDataUpid = steelKindData[steelKindData['upid_flag']==1].index
            goodData = data.loc[goodDataUpid][feature]
            badDataUpid = steelKindData[steelKindData['upid_flag']==0].index
            badData = data.loc[badDataUpid][feature]
            testBoardData = data.loc[upid][feature]

        
        # finalResult = []
            return {
                'goodData': goodData,
                'badData': badData,
                'testBoardData': testBoardData
            }
