'''
svrMethod
'''
# https://scikit-learn.org/stable/modules/svm.html#svr
from sklearn.svm import SVR as SupportVectorRegressor
import numpy as np

class SVR:
    '''
    SVR
    '''

    def __init__(self, custom_params):
        self.paramsIllegal = False
        try:
            self.C = float(custom_params['C'])
            self.epsilon = float(custom_params['epsilon'])
            self.degree = int(custom_params['degree'])
        except Exception:
            self.paramsIllegal = True

    def general_call(self, custom_input):
        '''
        general_call
        '''
        clf = SupportVectorRegressor(
            C=self.C,
            epsilon=self.epsilon,
            degree=self.degree,
            gamma='auto'
        )
        self.model = clf.fit(
            np.array(custom_input['X_train']).astype(np.float64),
            np.array(custom_input['y_train']).astype(np.float64)
        )
        return {
            'model': self.model
        }
