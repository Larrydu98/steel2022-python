'''
train_test_split
'''
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
import numpy as np

class TrainTestSplit:
    '''
    TrainTestSplit
    '''

    def __init__(self, custom_params):
        self.paramsIllegal = False
        try:
            self.test_size = float(custom_params['test_size'])
            self.shuffle = bool(custom_params['shuffle'])
        except Exception:
            self.paramsIllegal = True

    def general_call(self, custom_input):
        '''
        general_call
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(custom_input['X']).astype(np.float64),
            np.array(custom_input['y']).astype(np.float64),
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=42
        )
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

