'''
Standardization
'''
# https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
from sklearn.preprocessing import StandardScaler
import numpy as np

class Standardization:
    '''
    Standardization
    '''

    def __init__(self):
        self.paramsIllegal = False
        try:
            # 如果以后加入参数，在这里加入
            print('Standardization方法暂时没有参数')
        except Exception:
            self.paramsIllegal = True

    def general_call(self, custom_input):
        '''
        general_call
        '''
        self.model = StandardScaler().fit(np.array(custom_input['X']).astype(np.float64))
        array = self.model.transform(np.array(custom_input['X']).astype(np.float64))
        return {
            'array': array
        }
