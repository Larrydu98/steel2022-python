'''
deepCNNSinTestControllerNew
'''
import numpy as np
from flask import json
import pandas as pd
from  .getDataController import getDataController
from ..methods.algorithmChooseMethods.DeepCNN import DeepCNN
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import os
from ..staticPath import staticPath
from sklearn.metrics import roc_curve, auc
from keras import backend as K

class deepCNNSinTestController:
    '''
    deepCNNSinTestController
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
          
    def run(self, inputData,upid_someplate):
        '''
        run
        '''

        path = os.getcwd()
        # model_test = load_model('C:\pycharmProject\\flask-api\yykp\methods\\BS_Predict_WideAndDeep_FuMCcOutput117_1_Fufladc6_40_Pg3_100_MHpassMeas_Cctemperature2D9_100_p12456_5_100_MHpassPost_newFlag_batchsize8_300_10_3622_accmax_new0508.h5')
        model_test = load_model(path + staticPath.model_h5_w)
        # model_test = load_model(path + staticPath.model_h5_l)
        x_wide = np.array(inputData['x_wide'])
        x_deep1 = np.array(inputData['x_deep1'])
        x_deep2 = np.array(inputData['x_deep2'])
        x_deep3 = np.array(inputData['x_deep3'])
        x_deep4 = np.array(inputData['x_deep4'])
        x_deep5 = np.array(inputData['x_deep5'])
        x_deep6 = np.array(inputData['x_deep6'])
        y_2class = np.array(inputData['y_data2class'])
        y_1class = np.array(inputData['y_data1class'])


        K.clear_session()
        
        print('load model...')
        model_test = load_model(path + staticPath.model_h5_w)
        print('load done.')

        print('test model...')
        print(model_test.predict([np.zeros((1, x_wide.shape[1], x_wide.shape[2],x_wide.shape[3])), 
        np.zeros((1, x_deep1.shape[1],x_deep1.shape[2], x_deep1.shape[3])), 
        np.zeros((1, x_deep2.shape[1],x_deep2.shape[2], x_deep2.shape[3])), 
        np.zeros((1, x_deep3.shape[1],x_deep3.shape[2], x_deep3.shape[3])),
        np.zeros((1, x_deep4.shape[1],x_deep4.shape[2], x_deep4.shape[3])), 
        np.zeros((1, x_deep5.shape[1],x_deep5.shape[2], x_deep5.shape[3])), 
        np.zeros((1, x_deep6.shape[1],x_deep6.shape[2], x_deep6.shape[3]))]))
        print('test done.')


        upid_all_after_filter = pd.DataFrame(inputData['upid_np'].astype('str'),columns=['upid'])

        upid_all_before_filter = getDataController(6, '', '').run()

        upid_all_after_filter_drop = upid_all_after_filter.reset_index()
        upid_all_after_filter_drop = upid_all_after_filter_drop.drop(['index'], axis=1)


        upid_list = upid_someplate.upid.values.tolist()
        upid_flag_CR = upid_someplate.flag.values.tolist()

        # 对一块钢板数据进行预测
        result = []

        # print(upid_list)
        # print(upid_all_after_filter_drop)

        for upid in upid_list:

            upid_test_one = upid


            if upid_all_after_filter_drop[upid_all_after_filter_drop['upid'] == upid_test_one].empty:

                if upid_all_before_filter[upid_all_before_filter['upid'] == upid_test_one].empty:
                    # upid_flag_one_CR = []
                    # upid_flag_one_CR.append(upid_someplate[upid_someplate['upid'] == upid_test_one].flag.values[0])
                    # upid_flag_one_CR.append(1)
                    # result.append(upid_flag_one_CR)
                    #####0表示好，1表示不好
                    result.append(1-upid_someplate[upid_someplate['upid'] == upid_test_one].flag.values[0])


                else:
                    # upid_flag_one = []
                    # upid_flag_one.append(upid_all_before_filter[upid_all_before_filter['upid'] == upid_test_one].upid_flag.values[0])
                    # upid_flag_one.append(1)
                    # result.append(upid_flag_one)
                    #####0表示好，1表示不好
                    result.append(1-upid_all_before_filter[upid_all_before_filter['upid'] == upid_test_one].upid_flag.values[0])

            else:
                # print(2)
                test_index = upid_all_after_filter_drop[upid_all_after_filter_drop['upid'] == upid_test_one].index.values[0]
                x_test_wide_one = x_wide[test_index].reshape(1, x_wide.shape[1],x_wide.shape[2],x_wide.shape[3])
                x_test_deep1_one = x_deep1[test_index].reshape(1, x_deep1.shape[1],x_deep1.shape[2],x_deep1.shape[3])
                x_test_deep2_one = x_deep2[test_index].reshape(1, x_deep2.shape[1],x_deep2.shape[2],x_deep2.shape[3])
                x_test_deep3_one = x_deep3[test_index].reshape(1, x_deep3.shape[1],x_deep3.shape[2],x_deep3.shape[3])
                x_test_deep4_one = x_deep4[test_index].reshape(1, x_deep4.shape[1],x_deep4.shape[2],x_deep4.shape[3])
                x_test_deep5_one = x_deep5[test_index].reshape(1, x_deep5.shape[1],x_deep5.shape[2],x_deep5.shape[3])
                x_test_deep6_one = x_deep6[test_index].reshape(1, x_deep6.shape[1],x_deep6.shape[2],x_deep6.shape[3])
                y_test_one = y_1class[test_index]

                model_test_predict_one = model_test.predict([x_test_wide_one, x_test_deep1_one, x_test_deep2_one, x_test_deep3_one,
                     x_test_deep4_one, x_test_deep5_one, x_test_deep6_one])

                # model_test_predict_one_list = []
                # y_predict_oneclass = np.argmax(model_test_predict_one, axis=1)
                # model_test_predict_one_list.append(y_predict_oneclass[0])
                # model_test_predict_one_list.append(model_test_predict_one[0][y_predict_oneclass[0]])
                # result.append(model_test_predict_one_list)
                #####0表示好，1表示不好
                result.append(model_test_predict_one[0][0])
        #####0表示好，1表示不好
        result = np.array(result).tolist()
  
        return result


class deepCNNOnePlatePredictController:
    '''
    deepCNNOnePlatePredictController
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

    def run(self, inputData, upid_one):
        '''
        run
        '''
        
        path = os.getcwd()
        # model_test = load_model('C:\pycharmProject\\flask-api\yykp\methods\\BS_Predict_WideAndDeep_FuMCcOutput117_1_Fufladc6_40_Pg3_100_MHpassMeas_Cctemperature2D9_100_p12456_5_100_MHpassPost_newFlag_batchsize8_300_10_3622_accmax_new0508.h5')

        
        # model_test = load_model(path + staticPath.model_h5_l)
        x_wide = np.array(inputData['x_wide'])
        x_deep1 = np.array(inputData['x_deep1'])
        x_deep2 = np.array(inputData['x_deep2'])
        x_deep3 = np.array(inputData['x_deep3'])
        x_deep4 = np.array(inputData['x_deep4'])
        x_deep5 = np.array(inputData['x_deep5'])
        x_deep6 = np.array(inputData['x_deep6'])
        y_2class = np.array(inputData['y_data2class'])
        y_1class = np.array(inputData['y_data1class'])


        # x_test_wide_test = x_wide[1].reshape(1, x_wide.shape[1], x_wide.shape[2],x_wide.shape[3])
        # x_test_deep1_test = x_deep1[1].reshape(1, x_deep1.shape[1],x_deep1.shape[2], x_deep1.shape[3])
        # x_test_deep2_test = x_deep2[1].reshape(1, x_deep2.shape[1],x_deep2.shape[2], x_deep2.shape[3])
        # x_test_deep3_test = x_deep3[1].reshape(1, x_deep3.shape[1],x_deep3.shape[2], x_deep3.shape[3])
        # x_test_deep4_test = x_deep4[1].reshape(1, x_deep4.shape[1],x_deep4.shape[2], x_deep4.shape[3])
        # x_test_deep5_test = x_deep5[1].reshape(1, x_deep5.shape[1],x_deep5.shape[2], x_deep5.shape[3])
        # x_test_deep6_test = x_deep6[1].reshape(1, x_deep6.shape[1],x_deep6.shape[2], x_deep6.shape[3])


        K.clear_session()

        print('load model...')
        model_test = load_model(path + staticPath.model_h5_w)
        print('load done.')

        print('test model...')
        print(model_test.predict([np.zeros((1, x_wide.shape[1], x_wide.shape[2],x_wide.shape[3])), 
        np.zeros((1, x_deep1.shape[1],x_deep1.shape[2], x_deep1.shape[3])), 
        np.zeros((1, x_deep2.shape[1],x_deep2.shape[2], x_deep2.shape[3])), 
        np.zeros((1, x_deep3.shape[1],x_deep3.shape[2], x_deep3.shape[3])),
        np.zeros((1, x_deep4.shape[1],x_deep4.shape[2], x_deep4.shape[3])), 
        np.zeros((1, x_deep5.shape[1],x_deep5.shape[2], x_deep5.shape[3])), 
        np.zeros((1, x_deep6.shape[1],x_deep6.shape[2], x_deep6.shape[3]))]))
        print('test done.')


        upid_all_after_filter = pd.DataFrame(inputData['upid_np'].astype('str'), columns=['upid'])

        upid_all_after_filter_drop = upid_all_after_filter.reset_index()
        upid_all_after_filter_drop = upid_all_after_filter_drop.drop(['index'], axis=1)

        # 对一块钢板数据进行预测

        if upid_all_after_filter_drop[upid_all_after_filter_drop['upid'] == upid_one].empty:

            print('钢板数据不全')
            result = '钢板数据不全'
        else:

            test_index = upid_all_after_filter_drop[upid_all_after_filter_drop['upid'] == upid_one].index.values[0]
            x_test_wide_one = x_wide[test_index].reshape(1, x_wide.shape[1], x_wide.shape[2],x_wide.shape[3])
            x_test_deep1_one = x_deep1[test_index].reshape(1, x_deep1.shape[1],x_deep1.shape[2], x_deep1.shape[3])
            x_test_deep2_one = x_deep2[test_index].reshape(1, x_deep2.shape[1],x_deep2.shape[2], x_deep2.shape[3])
            x_test_deep3_one = x_deep3[test_index].reshape(1, x_deep3.shape[1],x_deep3.shape[2], x_deep3.shape[3])
            x_test_deep4_one = x_deep4[test_index].reshape(1, x_deep4.shape[1],x_deep4.shape[2], x_deep4.shape[3])
            x_test_deep5_one = x_deep5[test_index].reshape(1, x_deep5.shape[1],x_deep5.shape[2], x_deep5.shape[3])
            x_test_deep6_one = x_deep6[test_index].reshape(1, x_deep6.shape[1],x_deep6.shape[2], x_deep6.shape[3])
            y_test_one = str(y_1class[test_index])

            model_test_predict_one = model_test.predict([x_test_wide_one, x_test_deep1_one, x_test_deep2_one, x_test_deep3_one,
                     x_test_deep4_one, x_test_deep5_one, x_test_deep6_one])

            model_test_predict_one = np.around(model_test_predict_one, decimals=4)*100

            result = {'model_predict_one': model_test_predict_one[0].tolist(),
                      'y_test_one': y_test_one}

        return result


class deepCNNPlatePredictByTimeController:
    '''
    deepCNNPlatePredictByTimeController
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
        # model_test = load_model('C:\pycharmProject\\flask-api\yykp\methods\\BS_Predict_WideAndDeep_FuMCcOutput117_1_Fufladc6_40_Pg3_100_MHpassMeas_Cctemperature2D9_100_p12456_5_100_MHpassPost_newFlag_batchsize8_300_10_3622_accmax_new0508.h5')

        # model_test = load_model(path + staticPath.model_h5_w)

        x_wide = np.array(inputData['x_wide'])
        x_deep1 = np.array(inputData['x_deep1'])
        x_deep2 = np.array(inputData['x_deep2'])
        x_deep3 = np.array(inputData['x_deep3'])
        x_deep4 = np.array(inputData['x_deep4'])
        x_deep5 = np.array(inputData['x_deep5'])
        x_deep6 = np.array(inputData['x_deep6'])
        y_2class = np.array(inputData['y_data2class'])
        y_1class = np.array(inputData['y_data1class'])

        upid_all_after_filter = inputData['upid_np'].astype('str')

        K.clear_session()

        print('load model...')
        model_test = load_model(path + staticPath.model_h5_w)
        print('load done.')
        
        print('test model...')
        print(model_test.predict([np.zeros((1, x_wide.shape[1], x_wide.shape[2],x_wide.shape[3])), 
        np.zeros((1, x_deep1.shape[1],x_deep1.shape[2], x_deep1.shape[3])), 
        np.zeros((1, x_deep2.shape[1],x_deep2.shape[2], x_deep2.shape[3])), 
        np.zeros((1, x_deep3.shape[1],x_deep3.shape[2], x_deep3.shape[3])),
        np.zeros((1, x_deep4.shape[1],x_deep4.shape[2], x_deep4.shape[3])), 
        np.zeros((1, x_deep5.shape[1],x_deep5.shape[2], x_deep5.shape[3])), 
        np.zeros((1, x_deep6.shape[1],x_deep6.shape[2], x_deep6.shape[3]))]))
        print('test done.')


        # 对一段时间的钢板数据进行预测

        loss, acc = model_test.evaluate([x_wide, x_deep1, x_deep2, x_deep3, x_deep4, x_deep5,x_deep6], y_2class, batch_size=1)

        model_test_predict = model_test.predict([x_wide, x_deep1, x_deep2, x_deep3, x_deep4, x_deep5,x_deep6])

        model_test_predict_byTime = np.array(model_test_predict)
        y_predict = np.argmax(model_test_predict_byTime, axis=1)
        matr = confusion_matrix(y_1class, y_predict)

        # 计算概率正确的正负曲线图

        index_max = np.argmax(model_test_predict_byTime, axis=1)
        predict_max = model_test_predict[range(model_test_predict.shape[0]), index_max]
        predict_ture_flase = predict_max * (((index_max == y_1class) - 1) * 3 + ((index_max != y_1class) + 1))


        # 计算ROC曲线与AUC值

        fpr, tpr, threshold = roc_curve(y_1class, model_test_predict_byTime[:, 1])  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值

        loss = np.around(loss,decimals=4)
        acc = np.around(acc, decimals=4)

        model_test_predict_byTime = np.around(model_test_predict_byTime, decimals=4)

        result = {'loss': loss, 'acc': acc,
                  'fpr': fpr.tolist(),
                  'tpr': tpr.tolist(),
                  'roc_auc': roc_auc,
                  'model_test_predict': model_test_predict_byTime.tolist(),
                  'predict_ture_flase': predict_ture_flase.tolist(),
                  'y_predict': y_predict.tolist(),
                  'y_1class': y_1class.tolist(),
                  'matr': matr.tolist(),
                  'upid': upid_all_after_filter.tolist()}


        return result