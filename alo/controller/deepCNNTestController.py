'''
deepCNNTestController
'''
import numpy as np
from flask import json
from ..methods.algorithmChooseMethods.DeepCNN import DeepCNN
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import os
from ..staticPath import staticPath
from keras import backend as K

class deepCNNTestController:
    '''
    deepCNNTestController
    '''
    def __init__(self):
        self.idIllegal = False

    # 创建模型实例   
    def createModel(self):
        '''
        createModel
        '''
        model = DeepCNN()
        return model
          
    def run(self, inputData):
        '''
        run
        '''
        path = os.getcwd()
        x_test_wide_best = np.array(inputData['x_test_wide'])
        x_test_deep1_best = np.array(inputData['x_test_deep1'])
        x_test_deep2_best = np.array(inputData['x_test_deep2'])
        x_test_deep3_best = np.array(inputData['x_test_deep3'])
        x_test_deep4_best = np.array(inputData['x_test_deep4'])
        x_test_deep5_best = np.array(inputData['x_test_deep5'])
        x_test_deep6_best = np.array(inputData['x_test_deep6'])
        y_test_best = np.array(inputData['y_test'])
        y_test_raw = np.array(inputData['y_test_raw'])
        # model_dict_history = model_dict_history   ####改一下
        upid_train = inputData['upid_train_np']
        upid_test = inputData['upid_test_np']
        # model_test = self.createModel()
        # model_test = load_model('C:\pycharmProject\\flask-api\yykp\methods\\BS_Predict_WideAndDeep_FuMCcOutput117_1_Fufladc6_40_Pg3_100_MHpassMeas_Cctemperature2D9_100_p12456_5_100_MHpassPost_newFlag_batchsize8_300_10_3622_accmax_new0508.h5')

        # model_test = load_model(path + staticPath.model_h5_w)


        K.clear_session()
        
        print('load model...')
        model_test = load_model(path + staticPath.model_h5_w)
        print('load done.')
        
        print('test model...')
        print(model_test.predict([np.zeros((1, x_test_wide_best.shape[1], x_test_wide_best.shape[2],x_test_wide_best.shape[3])), 
        np.zeros((1, x_test_deep1_best.shape[1],x_test_deep1_best.shape[2], x_test_deep1_best.shape[3])), 
        np.zeros((1, x_test_deep2_best.shape[1],x_test_deep2_best.shape[2], x_test_deep2_best.shape[3])), 
        np.zeros((1, x_test_deep3_best.shape[1],x_test_deep3_best.shape[2], x_test_deep3_best.shape[3])),
        np.zeros((1, x_test_deep4_best.shape[1],x_test_deep4_best.shape[2], x_test_deep4_best.shape[3])), 
        np.zeros((1, x_test_deep5_best.shape[1],x_test_deep5_best.shape[2], x_test_deep5_best.shape[3])), 
        np.zeros((1, x_test_deep6_best.shape[1],x_test_deep6_best.shape[2], x_test_deep6_best.shape[3]))]))
        print('test done.')

        # 测试集预测结果
        model_test_predict = model_test.predict([x_test_wide_best, x_test_deep1_best, x_test_deep2_best, x_test_deep3_best, x_test_deep4_best, x_test_deep5_best,x_test_deep6_best])
        
        loss, acc = model_test.evaluate([x_test_wide_best, x_test_deep1_best, x_test_deep2_best, x_test_deep3_best, x_test_deep4_best, x_test_deep5_best,x_test_deep6_best], y_test_best, batch_size=1)

        # 模型迭代过程结果

        with open(path + staticPath.model_json_w, 'r') as load_f:
        # with open(path + staticPath.model_json_l, 'r') as load_f:
            model_dict_history = json.load(load_f)

        train_loss = model_dict_history['loss']
        verification_loss = model_dict_history['val_loss']

        train_acc = model_dict_history['acc']
        verification_acc = model_dict_history['val_acc']

        # 计算混淆矩阵
        y_predict = np.argmax(model_test_predict, axis=1)
        matr = confusion_matrix(y_test_raw, y_predict)

        # 计算权重
        weight_Dense_1, bias_Dense_1 = model_test.get_layer('dense_116').get_weights()
        weight_Dense_1_np = np.array(weight_Dense_1)
        weight_Dense_1_np = np.abs(weight_Dense_1_np)

        wide_weight = weight_Dense_1_np[0:64]
        deep1_weight = weight_Dense_1_np[len(wide_weight):len(wide_weight) + 256]
        deep2_weight = weight_Dense_1_np[len(wide_weight) + len(deep1_weight):len(wide_weight) + len(deep1_weight) + 512]
        deep3_weight = weight_Dense_1_np[len(wide_weight) + len(deep1_weight) + len(deep2_weight):len(wide_weight) + len(deep1_weight) + len(deep2_weight) + 512]
        deep4_weight = weight_Dense_1_np[len(wide_weight) + len(deep1_weight) + len(deep2_weight) + len(deep3_weight):len(wide_weight) + len(deep1_weight) + len(deep2_weight) + len(deep3_weight) + 512]
        deep5_weight = weight_Dense_1_np[len(wide_weight) + len(deep1_weight) + len(deep2_weight) + len(deep3_weight) + len(deep4_weight):len(wide_weight) + len(
                           deep1_weight) + len(deep2_weight) + len(deep3_weight) + len(deep4_weight) + 512]
        deep6_weight = weight_Dense_1_np[len(wide_weight) + len(deep1_weight) + len(deep2_weight) + len(deep3_weight) + len(deep4_weight) + len(deep5_weight):len(
                           wide_weight) + len(deep1_weight) + len(deep2_weight) + len(deep3_weight) + len(deep4_weight) + len(deep5_weight) + 512]

        wide_weight_sum = np.sum(wide_weight)/ 64
        deep1_weight_sum = np.sum(deep1_weight)/ 256
        deep2_weight_sum = np.sum(deep2_weight)/ 512
        deep3_weight_sum = np.sum(deep3_weight)/ 512
        deep4_weight_sum = np.sum(deep4_weight)/ 512
        deep5_weight_sum = np.sum(deep5_weight)/ 512
        deep6_weight_sum = np.sum(deep6_weight)/ 512

        Model_Weights_list_ave = []
        Model_Weights_list = [wide_weight_sum,deep1_weight_sum,deep2_weight_sum,deep3_weight_sum,deep4_weight_sum,deep5_weight_sum,deep6_weight_sum]

        Model_Weights_list_ave.append(wide_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep1_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep2_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep3_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep4_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep5_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep6_weight_sum / np.sum(np.array(Model_Weights_list)))

        loss = np.around(loss,decimals=2)
        acc = np.around(acc, decimals=4)
        train_loss = np.around(np.array(train_loss), decimals=2)
        train_acc = np.around(np.array(train_acc), decimals=4)
        verification_loss = np.around(np.array(verification_loss), decimals=2)
        verification_acc = np.around(np.array(verification_acc), decimals=4)
        model_test_predict = np.around(model_test_predict, decimals=2)

        result = {
            'upid_train': upid_train.tolist(),
            'upid_test': upid_test.tolist(),
            'loss': loss.tolist(),'acc': acc.tolist(),
            'train_loss': train_loss.tolist(),
            'verification_loss': verification_loss.tolist(),
            'train_acc': train_acc.tolist(),
            'verification_acc': verification_acc.tolist(),
            'model_test_predict': model_test_predict.tolist(),
            'y_predict': y_predict.tolist(),
            'matr': matr.tolist(),
            'Model_Weights_list_ave': Model_Weights_list_ave}
        
        return result


class getModelWeightsController:
    '''
    getModelWeightsController
    '''

    def __init__(self):
        self.idIllegal = False

    # 创建模型实例
    def createModel(self):
        '''
        createModel
        '''
        model = DeepCNN()
        return model

    def run(self):
        '''
        run
        '''
        path = os.getcwd()

        model_test = load_model(path + staticPath.model_h5_w)

        # 计算权重
        weight_Dense_1, bias_Dense_1 = model_test.get_layer('dense_116').get_weights()
        weight_Dense_1_np = np.array(weight_Dense_1)
        weight_Dense_1_np = np.abs(weight_Dense_1_np)

        wide_weight = weight_Dense_1_np[0:64]
        deep1_weight = weight_Dense_1_np[len(wide_weight):len(wide_weight) + 256]
        deep2_weight = weight_Dense_1_np[len(wide_weight) + len(deep1_weight):len(wide_weight) + len(deep1_weight) + 512]
        deep3_weight = weight_Dense_1_np[len(wide_weight) + len(deep1_weight) + len(deep2_weight):len(wide_weight) + len(deep1_weight) + len(deep2_weight) + 512]
        deep4_weight = weight_Dense_1_np[len(wide_weight) + len(deep1_weight) + len(deep2_weight) + len(deep3_weight):len(wide_weight) + len(deep1_weight) + len(
                           deep2_weight) + len(deep3_weight) + 512]
        deep5_weight = weight_Dense_1_np[len(wide_weight) + len(deep1_weight) + len(deep2_weight) + len(deep3_weight) + len(deep4_weight):len(wide_weight) + len(
                           deep1_weight) + len(deep2_weight) + len(deep3_weight) + len(deep4_weight) + 512]
        deep6_weight = weight_Dense_1_np[len(wide_weight) + len(deep1_weight) + len(deep2_weight) + len(deep3_weight) + len(deep4_weight) + len(deep5_weight):len(
                           wide_weight) + len(deep1_weight) + len(deep2_weight) + len(deep3_weight) + len(deep4_weight) + len(deep5_weight) + 512]

        wide_weight_sum = np.sum(wide_weight) / 64
        deep1_weight_sum = np.sum(deep1_weight) / 256
        deep2_weight_sum = np.sum(deep2_weight) / 512
        deep3_weight_sum = np.sum(deep3_weight) / 512
        deep4_weight_sum = np.sum(deep4_weight) / 512
        deep5_weight_sum = np.sum(deep5_weight) / 512
        deep6_weight_sum = np.sum(deep6_weight) / 512

        Model_Weights_list_ave = []
        Model_Weights_list = [wide_weight_sum, deep1_weight_sum, deep2_weight_sum, deep3_weight_sum,deep4_weight_sum, deep5_weight_sum, deep6_weight_sum]

        Model_Weights_list_ave.append(wide_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep1_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep2_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep3_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep4_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep5_weight_sum / np.sum(np.array(Model_Weights_list)))
        Model_Weights_list_ave.append(deep6_weight_sum / np.sum(np.array(Model_Weights_list)))


        result =  Model_Weights_list_ave

        return result
