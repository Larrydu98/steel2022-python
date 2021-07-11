'''
gbrMethod
'''
# https://scikit-learn.org/stable/modules/ensemble.html#regression
from sklearn.ensemble import GradientBoostingRegressor as GBR
import numpy as np

class GradientBoostingRegressor:
    '''
    GradientBoostingRegressor
    '''

    def __init__(self, custom_params):
        self.paramsIllegal = False
        try:
            self.max_depth = int(custom_params['max_depth'])
        except Exception:
            self.paramsIllegal = True
    
    def general_call(self, custom_input):
        '''
        general_call
        '''
        clf = GBR(max_depth=self.max_depth)
        self.model = clf.fit(
            np.array(custom_input['X_train']).astype(np.float64),
            np.array(custom_input['y_train']).astype(np.float64)
        )
        return {
            'model': self.model
        }
