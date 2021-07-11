'''
DeepCNN
'''
import os
from ...staticPath import staticPath
class DeepCNN:
    '''
    DeepCNN
    '''

    def __init__(self):
        self.paramsIllegal = False
        try:
            print('这个方法暂时没有参数')
        except Exception:
            self.paramsIllegal = True 

    def general_call():
         # 加载模型
        path = os.getcwd() 
        model_test = load_model(path + staticPath.model_h5_w)
        # model_test = load_model(path + staticPath.model_h5_l)
        return model_test
