'''
createDiagResu
'''
# https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
import pandas as pd
import numpy as np
import traceback
import json
from sklearn import preprocessing
from scipy.stats import f
from scipy.stats import norm
from ...utils import getData_bytime
from ...api.singelSteel import data_names, without_cooling_data_names, data_names_meas
from ...models.monitorResuModel import getMonitorTrainData
import copy
import datetime


class PCATEST:

    '''
    PCATEST
    '''

    def __init__(self):
        self.paramsIllegal = False

        try:
            # 如果以后加入参数，在这里加入
            # print('PCATEST方法暂时没有参数')
            pass
        except Exception:
            self.paramsIllegal = True

    def general_call(self, custom_input):
        '''
        general_call
        '''
        Xtrain = custom_input['Xtrain']
        Xtest  = custom_input['Xtest']
        X_row  = Xtrain.shape[0]
        X_col  = Xtrain.shape[1]
        X_mean = np.mean(Xtrain, axis=0)
        X_std  = np.std(Xtrain, axis=0)
        Xtrain = (Xtrain - np.tile(X_mean, (X_row, 1))) / np.tile(X_std, (X_row, 1))
        
        sigmaXtrain = np.cov(Xtrain.T)
        [lamda, T] = np.linalg.eigh(sigmaXtrain)
        num_pc = 1
        D = -np.sort(-lamda, axis=0)
        lamda = np.diag(lamda)
        while D[0:num_pc].sum(axis=0) / D.sum(axis=0) < 0.9:
            num_pc = num_pc + 1
        P = T[:, np.arange(X_col - num_pc, X_col)]
        T2UCL1 = num_pc * (X_row - 1) * (X_row + 1) * f.ppf(0.99, num_pc, X_row - num_pc) / (X_row * (X_row - num_pc))
        T2UCL2 = num_pc * (X_row - 1) * (X_row + 1) * f.ppf(0.95, num_pc, X_row - num_pc) / (X_row * (X_row - num_pc))
        theta = np.zeros(3)
        for i in range(3):
            theta[i] = np.sum((D[np.arange(num_pc, X_col)]) ** (i + 1))
        h0 = 1 - 2 * theta[0] * theta[2] / (3 * theta[1] ** 2)
        ca = norm.ppf(0.99, 0, 1)
        QUCL = theta[0] * (
                    h0 * ca * np.sqrt(2. * theta[1]) / theta[0] + 1 + theta[1] * h0 * (h0 - 1.) / theta[0] ** 2.) ** (
                        1. / h0)
        n = Xtest.shape[0]
        m = Xtest.shape[1]

        Xtest = (Xtest - np.tile(X_mean, (n, 1))) / np.tile(X_std, (n, 1))
        P = np.matrix(P)
        [r, y] = (P * P.T).shape
        I = np.eye(r, y)
        T2 = np.zeros((n, 1))
        Q = np.zeros((n, 1))
        for i in range(n):
            T2[i] = np.matrix(Xtest[i, :]) * P * np.matrix(
                (lamda[np.ix_(np.arange(m - num_pc, m), np.arange(m - num_pc, m))])).I * P.T * np.matrix(Xtest[i, :]).T
            
            Q[i] = np.matrix(Xtest[i, :]) * (I - P * P.T) * np.matrix(Xtest[i, :]).T
            
        test_Num = 0
        S = np.array(np.matrix(Xtest[test_Num, :]) * P[:, np.arange(0, num_pc)])
        S = S[0]
        r = []
        for i in range(num_pc):
            if S[i] ** 2 / lamda[i, 0] > T2UCL1 / num_pc:
                r.append(i)
        cont = np.zeros((len(r), m))
        for i in [len(r) - 1]:
            for j in range(m):
                cont[i][j] = np.fabs(S[i] / D[i] * P[j, i] * Xtest[test_Num, j])

        CONTJ = []
        for j in range(m):
            CONTJ.append(np.sum(cont[:, j]))
        e = np.matrix(Xtest[test_Num, :]) * (I - P * P.T)
        e = np.array(e)[0]
        contq = e ** 2
        return T2UCL1, T2UCL2, QUCL, T2, Q, CONTJ, contq

    def stage_general_call(self, custom_input):
        '''
        stage_general_call
        '''
        Xtrain = custom_input['Xtrain']
        Xtest = custom_input['Xtest']
        X_row = Xtrain.shape[0]
        X_col = Xtrain.shape[1]
        X_mean = np.mean(Xtrain, axis=0)
        X_std = np.std(Xtrain, axis=0)
        Xtrain = (Xtrain - np.tile(X_mean, (X_row, 1))) / np.tile(X_std, (X_row, 1))

        sigmaXtrain = np.cov(Xtrain.T)
        [lamda, T] = np.linalg.eigh(sigmaXtrain)
        num_pc = 1
        D = -np.sort(-lamda, axis=0)
        lamda = np.diag(lamda)
        while D[0:num_pc].sum(axis=0) / D.sum(axis=0) < 0.9:
            num_pc = num_pc + 1
        P = T[:, np.arange(X_col - num_pc, X_col)]
        T2UCL1 = num_pc * (X_row - 1) * (X_row + 1) * f.ppf(0.99, num_pc, X_row - num_pc) / (X_row * (X_row - num_pc))
        T2UCL2 = num_pc * (X_row - 1) * (X_row + 1) * f.ppf(0.95, num_pc, X_row - num_pc) / (X_row * (X_row - num_pc))
        theta = np.zeros(3)
        for i in range(3):
            theta[i] = np.sum((D[np.arange(num_pc, X_col)]) ** (i + 1))
        h0 = 1 - 2 * theta[0] * theta[2] / (3 * theta[1] ** 2)
        ca = norm.ppf(0.99, 0, 1)
        QUCL = theta[0] * (
                h0 * ca * np.sqrt(2. * theta[1]) / theta[0] + 1 + theta[1] * h0 * (h0 - 1.) / theta[0] ** 2.) ** (
                       1. / h0)
        n = Xtest.shape[0]
        m = Xtest.shape[1]

        Xtest = (Xtest - np.tile(X_mean, (n, 1))) / np.tile(X_std, (n, 1))
        P = np.matrix(P)
        [r, y] = (P * P.T).shape
        I = np.eye(r, y)
        T2 = np.zeros((n, 1))
        Q = np.zeros((n, 1))
        for i in range(n):
            T2[i] = np.matrix(Xtest[i, :]) * P * np.matrix(
                (lamda[np.ix_(np.arange(m - num_pc, m), np.arange(m - num_pc, m))])).I * P.T * np.matrix(Xtest[i, :]).T
            Q[i] = np.matrix(Xtest[i, :]) * (I - P * P.T) * np.matrix(Xtest[i, :]).T
        return T2UCL1, QUCL, T2, Q


def unidimensional_monitoring(upid_data_df, good_data_df, col_names, data_names_meas, quantile_num, extremum_quantile_num, s_extremum_quantile_num):
    ## 获取同一规格钢板的取过程数据
    process_data = good_data_df[col_names]
    process_data = process_data.drop(columns = data_names_meas).values

    good_meas_data = good_data_df[data_names_meas]
    upid_meas_data = upid_data_df[data_names_meas]
    good_meas_data = good_meas_data[good_meas_data.sum(axis = 1) > 0].values

    ## 计算原始过程数据的上下分位点
    lower_limit = np.quantile(process_data, quantile_num, axis=0)
    upper_limit = np.quantile(process_data, 1-quantile_num, axis=0)
    ## 计算原始过程数据的上下极值分位点
    extremum_lower_limit = np.quantile(process_data, extremum_quantile_num, axis=0)
    extremum_upper_limit = np.quantile(process_data, 1-extremum_quantile_num, axis=0)
    ## 计算原始过程数据的上下超级极值分位点
    s_extremum_lower_limit = np.quantile(process_data, s_extremum_quantile_num, axis=0)
    s_extremum_upper_limit = np.quantile(process_data, 1-s_extremum_quantile_num, axis=0)
    ## 查询此钢板的过程数据（若前端在点击马雷图或散点图前已经储存该数据，则此步骤可以省略）
    upid_data = upid_data_df.values[0]
    ## 对同一规格钢板过程数据进行归一化计算
    norm_process_data = ((process_data - process_data.min()) / (process_data.max() - process_data.min()))#.fillna(0)
    ## 计算归一化后过程数据的上下分位点
    lower_limit_norm = np.quantile(norm_process_data, quantile_num, axis=0)
    upper_limit_norm = np.quantile(norm_process_data, 1 - quantile_num, axis=0)
    ## 计算归一化后过程数据的上下极值分位点
    extremum_lower_limit_norm = np.quantile(norm_process_data, extremum_quantile_num, axis=0)
    extremum_upper_limit_norm = np.quantile(norm_process_data, 1 - extremum_quantile_num, axis=0)
    ## 计算归一化后过程数据的上下超级极值分位点
    s_extremum_lower_limit_norm = np.quantile(norm_process_data, s_extremum_quantile_num, axis=0)
    s_extremum_upper_limit_norm = np.quantile(norm_process_data, 1 - s_extremum_quantile_num, axis=0)
    ## 查询此钢板归一化后的过程数据
    # norm_upid_data = norm_process_data[good_data_df.upid == upid][col_names].values[0]
    norm_upid_data = ((upid_data_df - process_data.min()) / (process_data.max() - process_data.min())).fillna(0).values[0]
    norm_upid_data[norm_upid_data > 1] = 1
    norm_upid_data[norm_upid_data < 0] = 0
    # ## 计算归一化后超限幅度
    # over_limit_range = copy.copy(norm_upid_data)
    # extremum_over_limit_range = copy.copy(norm_upid_data)
    # s_extremum_over_limit_range = copy.copy(norm_upid_data)
    #
    # over_limit_range[np.where(norm_upid_data > upper_limit_norm)] = norm_upid_data[norm_upid_data > upper_limit_norm] - upper_limit_norm[norm_upid_data > upper_limit_norm]
    # over_limit_range[np.where(norm_upid_data < lower_limit_norm)] = norm_upid_data[norm_upid_data < lower_limit_norm] - lower_limit_norm[norm_upid_data < lower_limit_norm]
    # over_limit_range[np.where((norm_upid_data >= lower_limit_norm) & (norm_upid_data <= upper_limit_norm))] = 0
    #
    # extremum_over_limit_range[np.where(norm_upid_data > extremum_upper_limit_norm)] = norm_upid_data[norm_upid_data > extremum_upper_limit_norm] - extremum_upper_limit_norm[norm_upid_data > extremum_upper_limit_norm]
    # extremum_over_limit_range[np.where(norm_upid_data < extremum_lower_limit_norm)] = norm_upid_data[norm_upid_data < extremum_lower_limit_norm] - extremum_lower_limit_norm[norm_upid_data < extremum_lower_limit_norm]
    # extremum_over_limit_range[np.where((norm_upid_data >= extremum_lower_limit_norm) & (norm_upid_data <= extremum_upper_limit_norm))] = 0
    #
    # s_extremum_over_limit_range[np.where(norm_upid_data > s_extremum_upper_limit_norm)] = norm_upid_data[norm_upid_data > s_extremum_upper_limit_norm] - s_extremum_upper_limit_norm[norm_upid_data > s_extremum_upper_limit_norm]
    # s_extremum_over_limit_range[np.where(norm_upid_data < s_extremum_lower_limit_norm)] = norm_upid_data[norm_upid_data < s_extremum_lower_limit_norm] - s_extremum_lower_limit_norm[norm_upid_data < s_extremum_lower_limit_norm]
    # s_extremum_over_limit_range[np.where((norm_upid_data >= s_extremum_lower_limit_norm) & (norm_upid_data <= s_extremum_upper_limit_norm))] = 0
    meas_item_data = good_meas_data
    ## 计算原始过程数据的上下分位点
    meas_lower_limit = np.quantile(meas_item_data, quantile_num, axis=0)
    meas_upper_limit = np.quantile(meas_item_data, 1 - quantile_num, axis=0)
    ## 计算原始过程数据的上下极值分位点
    meas_extremum_lower_limit = np.quantile(meas_item_data, extremum_quantile_num, axis=0)
    meas_extremum_upper_limit = np.quantile(meas_item_data, 1 - extremum_quantile_num, axis=0)
    ## 计算原始过程数据的上下超级极值分位点
    meas_s_extremum_lower_limit = np.quantile(meas_item_data, s_extremum_quantile_num, axis=0)
    meas_s_extremum_upper_limit = np.quantile(meas_item_data, 1 - s_extremum_quantile_num, axis=0)
    ## 对同一规格钢板过程数据进行归一化计算
    meas_norm_process_data = ((meas_item_data - meas_item_data.min()) / (meas_item_data.max() - meas_item_data.min()))#.fillna(0)
    ## 计算归一化后过程数据的上下分位点
    meas_lower_limit_norm = np.quantile(meas_norm_process_data, quantile_num, axis=0)
    meas_upper_limit_norm = np.quantile(meas_norm_process_data, 1 - quantile_num, axis=0)
    ## 计算归一化后过程数据的上下极值分位点
    meas_extremum_lower_limit_norm = np.quantile(meas_norm_process_data, extremum_quantile_num, axis=0)
    meas_extremum_upper_limit_norm = np.quantile(meas_norm_process_data, 1 - extremum_quantile_num, axis=0)
    ## 计算归一化后过程数据的上下超级极值分位点
    meas_s_extremum_lower_limit_norm = np.quantile(meas_norm_process_data, s_extremum_quantile_num, axis=0)
    meas_s_extremum_upper_limit_norm = np.quantile(meas_norm_process_data, 1 - s_extremum_quantile_num, axis=0)
    ## 查询此钢板归一化后的过程数据
    meas_norm_upid_data = ((upid_meas_data - meas_item_data.min()) / (meas_item_data.max() - meas_item_data.min())).values[0]

    result = []
    proc_i = 0
    meas_i = 0
    for i in range(len(col_names)):
        if col_names[i] in data_names_meas:
            result.append({
                'name': col_names[i],

                'original_value': upid_data[i],
                'original_l': meas_lower_limit[meas_i],
                'original_u': meas_upper_limit[meas_i],
                'extremum_original_l': meas_extremum_lower_limit[meas_i],
                'extremum_original_u': meas_extremum_upper_limit[meas_i],
                's_extremum_original_l': meas_s_extremum_lower_limit[meas_i],
                's_extremum_original_u': meas_s_extremum_upper_limit[meas_i],

                'value': meas_norm_upid_data[meas_i],
                'l': meas_lower_limit_norm[meas_i],
                'u': meas_upper_limit_norm[meas_i],
                'extremum_l': meas_extremum_lower_limit_norm[meas_i],
                'extremum_u': meas_extremum_upper_limit_norm[meas_i],
                's_extremum_l': meas_s_extremum_lower_limit_norm[meas_i],
                's_extremum_u': meas_s_extremum_upper_limit_norm[meas_i]
            })
            meas_i += 1
        else:
            result.append({
                'name': col_names[i],

                'original_value': upid_data[i],
                'original_l': lower_limit[proc_i],
                'original_u': upper_limit[proc_i],
                'extremum_original_l': extremum_lower_limit[proc_i],
                'extremum_original_u': extremum_upper_limit[proc_i],
                's_extremum_original_l': s_extremum_lower_limit[proc_i],
                's_extremum_original_u': s_extremum_upper_limit[proc_i],

                'value': norm_upid_data[proc_i],
                'l': lower_limit_norm[proc_i],
                'u': upper_limit_norm[proc_i],
                'extremum_l': extremum_lower_limit_norm[proc_i],
                'extremum_u': extremum_upper_limit_norm[proc_i],
                's_extremum_l': s_extremum_lower_limit_norm[proc_i],
                's_extremum_u': s_extremum_upper_limit_norm[proc_i]
                })
            proc_i += 1
    return result


class createMonitorResu:
    '''
    createDiagResu
    '''

    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.paramsIllegal = False
        try:
            # 如果以后加入参数，在这里加入
            # print('createDiagResu方法暂时没有参数')
            pass
        except Exception:
            self.paramsIllegal = True

    def run(self, request_bodys, tocs, sorttype, limit):
        '''
        run
        '''
        monitor_result = {}
        batch_monitor_result = self.getBatchMonitorResult(request_bodys, tocs, sorttype, limit)
        for item in batch_monitor_result:
            monitor_result[item["upid"]] = item

        # 非合并的板的处理：
        nobatch_monitor_result = self.getNoBatchMonitorResult(monitor_result, request_bodys, tocs, sorttype, limit)
        for item in nobatch_monitor_result:
            monitor_result[item["upid"]] = item

        return monitor_result, 200

    def getBatchMonitorResult(self, request_bodys, tocs, sorttype, limit):
        monitor_result = []
        selection = ['dd.upid', 'lmpd.productcategory', 'dd.tgtwidth', 'dd.tgtlength', 'dd.tgtthickness', 'dd.stats', 'dd.fqc_label', 'dd.toc']
        for index, reqBody in enumerate(request_bodys):
            otherdata, _, status_cooling, fqcflag = self.getOtherData(reqBody, selection, limit)
            if len(otherdata) == 0:
                print('in batch' + str(index) + ', train data is less!')
                continue

            ismissing = {'status_stats': True,
                         'status_cooling': True if status_cooling == 0 else False,
                         'status_fqc': True if fqcflag == 0 else False}
            data, columns = getData_bytime(['upid', 'platetype', 'tgtwidth', 'tgtlength', 'tgtthickness', 'stats', 'fqc_label', 'toc'],
                                           ismissing, [], [], [],
                                           tocs[index], [], [], '', '')
            data_df = pd.DataFrame(data=data, columns=columns).dropna(axis=0, how='any').reset_index(drop=True)
            if len(data_df) == 0:
                print('in batch' + str(index) + ', test data is too less!')
                continue

            _data_names = self.getDataNames(status_cooling)
            goodBoardData, badBoardData, goodBoardId, badBoardId, labelArr, goodBoardDf = self.getBoardData(otherdata, _data_names, fqcflag)

            X_train = np.array(goodBoardData)
            X_zero_std = np.where((np.std(X_train, axis=0)) <= 1e-10)
            X_train = np.delete(X_train, X_zero_std, axis=1)
            if len(X_train) < 5:
                print('in batch' + str(index) + ', train data is too less!')
                continue

            for i in sorted(X_zero_std[0], reverse=True):
                del _data_names[i]
            X_Heat_train = X_train[:, 0:68]
            X_Roll_train = X_train[:, 68:97]
            if status_cooling == 0:
                X_Cool_train = X_train[:, 97:117]

            X_test = []
            for i in range(len(data)):
                one_process = []
                for data_name in _data_names:
                    one_process.append(data[i][5][data_name])
                one_process = list(map(lambda x: 0.0 if x is None else x, one_process))
                X_test.append(one_process)

            try:
                X_test = np.array(X_test)
                # X_test = np.delete(X_test, X_zero_std, axis=1)
                # T2UCL1, T2UCL2, QUCL, T2, Q, CONTJ, contq = PCATEST().general_call({
                #     'Xtrain': X_train,
                #     'Xtest': X_test,
                # })

                X_Heat_test = X_test[:, 0:68]
                X_Roll_test = X_test[:, 68:97]
                Heat_T2UCL, Heat_QUCL, Heat_T2, Heat_Q = PCATEST().stage_general_call({
                    'Xtrain': X_Heat_train,
                    'Xtest': X_Heat_test
                })
                Roll_T2UCL, Roll_QUCL, Roll_T2, Roll_Q = PCATEST().stage_general_call({
                    'Xtrain': X_Roll_train,
                    'Xtest': X_Roll_test
                })
                if status_cooling == 0:
                    X_Cool_test = X_test[:, 97:117]
                    Cool_T2UCL, Cool_QUCL, Cool_T2, Cool_Q = PCATEST().stage_general_call({
                        'Xtrain': X_Cool_train,
                        'Xtest': X_Cool_test
                    })
            except Exception:
                print('in batch' + str(index) + ', there are some error when computing PCA result!')
                print(traceback.format_exc())
                continue

            for i in range(len(data)):
                upid = data[i][0]
                platetype = data[i][1]
                tgtwidth = data[i][2]
                tgtlength = data[i][3]
                tgtthickness = data[i][4]
                try:
                    fqc_label = 1 if np.array(data[i][6]['method1']['data']).sum() == 5 else 0
                except:
                    fqc_label = 404
                toc = data[i][7]

                # result, CONTJ_Pro, contq_Pro = self.getDiagnosisResultForOnePlate(X_train, X_test[i, :], goodBoardDf, _data_names, data_names_meas)

                if status_cooling == 0:
                    monitor_result.append({
                        "upid": upid,
                        "toc": toc.strftime("%Y-%m-%d %H:%M:%S"),
                        "fqc_label": fqc_label,

                        # "T2UCL1": T2UCL1,
                        # "QUCL": QUCL,
                        # "T2": T2[i][0],
                        # "Q": Q[i][0],

                        # "one_dimens": result,
                        # "CONTJ": CONTJ_Pro,
                        # "CONTQ": contq_Pro,

                        "Heat_T2UCL": Heat_T2UCL,
                        "Heat_QUCL": Heat_QUCL,
                        "Heat_T2": Heat_T2[i][0],
                        "Heat_Q": Heat_Q[i][0],
                        "Roll_T2UCL": Roll_T2UCL,
                        "Roll_QUCL": Roll_QUCL,
                        "Roll_T2": Roll_T2[i][0],
                        "Roll_Q": Roll_Q[i][0],
                        "Cool_T2UCL": Cool_T2UCL,
                        "Cool_QUCL": Cool_QUCL,
                        "Cool_T2": Cool_T2[i][0],
                        "Cool_Q": Cool_Q[i][0]
                    })
                else:
                    monitor_result.append({
                        "upid": upid,
                        "toc": toc.strftime("%Y-%m-%d %H:%M:%S"),
                        "fqc_label": fqc_label,

                        # "T2UCL1": T2UCL1,
                        # "QUCL": QUCL,
                        # "T2": T2[i][0],
                        # "Q": Q[i][0],

                        # "one_dimens": result,
                        # "CONTJ": CONTJ_Pro,
                        # "CONTQ": contq_Pro,

                        "Heat_T2UCL": Heat_T2UCL,
                        "Heat_QUCL": Heat_QUCL,
                        "Heat_T2": Heat_T2[i][0],
                        "Heat_Q": Heat_Q[i][0],
                        "Roll_T2UCL": Roll_T2UCL,
                        "Roll_QUCL": Roll_QUCL,
                        "Roll_T2": Roll_T2[i][0],
                        "Roll_Q": Roll_Q[i][0]
                    })

        return monitor_result

    def getDiagnosisResultForOnePlate(self, X_train, X_test, goodBoardDf, data_names, data_names_meas):
        X_test =X_test.reshape((1, len(X_test)))
        T2UCL1, T2UCL2, QUCL, T2, Q, CONTJ, contq = PCATEST().general_call({
            'Xtrain': X_train,
            'Xtest': X_test,
        })

        CONTJ_Pro = []
        maxCON = max(CONTJ)
        minCON = min(CONTJ)
        for item in CONTJ:
            mid = (item - minCON) / (maxCON - minCON)
            CONTJ_Pro.append(mid)

        contq_Pro = []
        maxContq = max(contq.tolist())
        minContq = min(contq.tolist())
        for item in contq.tolist():
            mid = (item - minContq) / (maxContq - minContq)
            contq_Pro.append(mid)

        upid_data_df = pd.DataFrame(data=X_test, columns=data_names)
        result = unidimensional_monitoring(upid_data_df,
                                           goodBoardDf,
                                           data_names,
                                           data_names_meas,
                                           0.25, 0.05, 0.01)
        return result, CONTJ_Pro, contq_Pro


    def getNoBatchMonitorResult(self, _monitor_result, request_bodys, tocs, sorttype, limit):
        monitor_result = []
        selection = ['dd.upid', 'lmpd.productcategory', 'dd.tgtwidth', 'dd.tgtlength', 'dd.tgtthickness', 'dd.stats', 'dd.fqc_label', 'dd.toc', 'dd.status_cooling']
        noCooling_otherdata, hasCooling_otherdata = self.getNoBatchOtherData(request_bodys, selection, limit)
        ismissing = {'status_stats': True, 'status_fqc': True}
        data, columns = getData_bytime(['upid', 'platetype', 'tgtwidth', 'tgtlength', 'tgtthickness', 'stats', 'fqc_label', 'toc', 'status_cooling'],
                                       ismissing, [], [], [],
                                       [self.start_time, self.end_time], [], [], '', '')
        noCooling_data, hasCooling_data = [], []
        for item in data:
            try:
                _monitor_result[item[0]]
            except:
                if item[8] == 0:
                    hasCooling_data.append(item)
                elif item[8] == 1:
                    noCooling_data.append(item)
        hasCooling_data_names = self.getDataNames(0)
        noCooling_data_names = self.getDataNames(1)

        noCooling_StatsData, _, _, _, _, noCooling_goodBoardDf = self.getBoardData(noCooling_otherdata, noCooling_data_names, 1)
        noCooling_X_train = np.array(noCooling_StatsData)
        noCooling_X_zero_std = np.where((np.std(noCooling_X_train, axis=0)) <= 1e-10)
        noCooling_X_train = np.delete(noCooling_X_train, noCooling_X_zero_std, axis=1)
        for i in sorted(noCooling_X_zero_std[0], reverse=True):
            del noCooling_data_names[i]

        hasCooling_StatsData, _, _, _, _, hasCooling_goodBoardDf = self.getBoardData(hasCooling_otherdata, hasCooling_data_names, 1)
        hasCooling_X_train = np.array(hasCooling_StatsData)
        hasCooling_X_zero_std = np.where((np.std(hasCooling_X_train, axis=0)) <= 1e-10)
        hasCooling_X_train = np.delete(hasCooling_X_train, hasCooling_X_zero_std, axis=1)
        for i in sorted(hasCooling_X_zero_std[0], reverse=True):
            del hasCooling_data_names[i]

        self.computeNoBatchMoni(monitor_result, hasCooling_X_train, hasCooling_goodBoardDf, hasCooling_X_zero_std, hasCooling_data, hasCooling_data_names, 0)
        self.computeNoBatchMoni(monitor_result, noCooling_X_train, noCooling_goodBoardDf, noCooling_X_zero_std, noCooling_data, noCooling_data_names, 1)

        return monitor_result

    def computeNoBatchMoni(self, monitor_result, X_train, goodBoardDf, X_zero_std, data, _data_names, status_cooling):
        X_Heat_train = X_train[:, 0:68]
        X_Roll_train = X_train[:, 68:97]
        if status_cooling == 0:
            X_Cool_train = X_train[:, 97:117]

        X_test = []
        for i in range(len(data)):
            one_process = []
            for data_name in _data_names:
                one_process.append(data[i][5][data_name])
            one_process = list(map(lambda x: 0.0 if x is None else x, one_process))
            X_test.append(one_process)

        try:
            X_test = np.array(X_test)
            # X_test = np.delete(X_test, X_zero_std, axis=1)
            # T2UCL1, QUCL, T2, Q = PCATEST().stage_general_call({
            #     'Xtrain': X_train,
            #     'Xtest': X_test,
            # })

            X_Heat_test = X_test[:, 0:68]
            X_Roll_test = X_test[:, 68:97]
            Heat_T2UCL, Heat_QUCL, Heat_T2, Heat_Q = PCATEST().stage_general_call({
                'Xtrain': X_Heat_train,
                'Xtest': X_Heat_test
            })
            Roll_T2UCL, Roll_QUCL, Roll_T2, Roll_Q = PCATEST().stage_general_call({
                'Xtrain': X_Roll_train,
                'Xtest': X_Roll_test
            })
            if status_cooling == 0:
                X_Cool_test = X_test[:, 97:117]
                Cool_T2UCL, Cool_QUCL, Cool_T2, Cool_Q = PCATEST().stage_general_call({
                    'Xtrain': X_Cool_train,
                    'Xtest': X_Cool_test
                })
        except Exception:
            print(traceback.format_exc())
            return

        for i in range(len(data)):
            upid = data[i][0]
            platetype = data[i][1]
            tgtwidth = data[i][2]
            tgtlength = data[i][3]
            tgtthickness = data[i][4]
            try:
                fqc_label = 1 if np.array(data[i][6]['method1']['data']).sum() == 5 else 0
            except:
                fqc_label = 404
            toc = data[i][7]

            # result, CONTJ_Pro, contq_Pro = self.getDiagnosisResultForOnePlate(X_train, X_test[i, :], goodBoardDf, _data_names, data_names_meas)

            if status_cooling == 0:
                monitor_result.append({
                    "upid": upid,
                    "toc": toc.strftime("%Y-%m-%d %H:%M:%S"),
                    "fqc_label": fqc_label,

                    # "T2UCL1": T2UCL1,
                    # "QUCL": QUCL,
                    # "T2": T2[i][0],
                    # "Q": Q[i][0],

                    # "one_dimens": result,
                    # "CONTJ": CONTJ_Pro,
                    # "CONTQ": contq_Pro,

                    "Heat_T2UCL": Heat_T2UCL,
                    "Heat_QUCL": Heat_QUCL,
                    "Heat_T2": Heat_T2[i][0],
                    "Heat_Q": Heat_Q[i][0],
                    "Roll_T2UCL": Roll_T2UCL,
                    "Roll_QUCL": Roll_QUCL,
                    "Roll_T2": Roll_T2[i][0],
                    "Roll_Q": Roll_Q[i][0],
                    "Cool_T2UCL": Cool_T2UCL,
                    "Cool_QUCL": Cool_QUCL,
                    "Cool_T2": Cool_T2[i][0],
                    "Cool_Q": Cool_Q[i][0]
                })
            else:
                monitor_result.append({
                    "upid": upid,
                    "toc": toc.strftime("%Y-%m-%d %H:%M:%S"),
                    "fqc_label": fqc_label,

                    # "T2UCL1": T2UCL1,
                    # "QUCL": QUCL,
                    # "T2": T2[i][0],
                    # "Q": Q[i][0],

                    # "one_dimens": result,
                    # "CONTJ": CONTJ_Pro,
                    # "CONTQ": contq_Pro,

                    "Heat_T2UCL": Heat_T2UCL,
                    "Heat_QUCL": Heat_QUCL,
                    "Heat_T2": Heat_T2[i][0],
                    "Heat_Q": Heat_Q[i][0],
                    "Roll_T2UCL": Roll_T2UCL,
                    "Roll_QUCL": Roll_QUCL,
                    "Roll_T2": Roll_T2[i][0],
                    "Roll_Q": Roll_Q[i][0]
                })

    def getOtherData(self, reqBody, selection, limit):
        otherdata, _, status_cooling, fqcflag = getMonitorTrainData(reqBody, selection, limit)

        # 如果拿到的数据过少，则放宽指标的要求. 但是 是否过冷却 和 是否考虑标签不能去除
        if len(otherdata) <= 10:
            reqBody['steelspec'] = '[]'
            otherdata, _, status_cooling, fqcflag = getMonitorTrainData(reqBody, selection, limit)
            return otherdata, _, status_cooling, fqcflag
        return otherdata, _, status_cooling, fqcflag

    def getNoBatchOtherData(self, reqBody, selection, limit):
        noCooling_reqBody = copy.deepcopy(reqBody[0])
        noCooling_reqBody["steelspec"] = json.dumps([])
        noCooling_reqBody["status_cooling"] = "1"
        noCooling_reqBody["fqcflag"] = "0"
        noCooling_otherdata, _, _, _ = getMonitorTrainData(noCooling_reqBody, selection, limit)

        hasCooling_reqBody = copy.deepcopy(reqBody[0])
        hasCooling_reqBody["steelspec"] = json.dumps([])
        hasCooling_reqBody["status_cooling"] = "0"
        hasCooling_reqBody["fqcflag"] = "0"
        hasCooling_otherdata, _, _, _ = getMonitorTrainData(hasCooling_reqBody, selection, limit)

        return noCooling_otherdata, hasCooling_otherdata

    def getDataNames(self, status_cooling):
        if status_cooling == 0:
            return copy.deepcopy(data_names)
        elif status_cooling == 1:
            return copy.deepcopy(without_cooling_data_names)

    def getBoardData(self, otherData, data_names, fqcflag):
        labelArr = []
        badBoardData = []
        goodBoardData = []
        badBoardId = []
        goodBoardId = []

        if fqcflag == 0:
            for item in otherData:
                flags = item[6]['method1']['data']
                label = 0
                if np.array(flags).sum() == 5:
                    label = 1
                labelArr.append(label)

                item_data = []
                # print(item[0])
                for data_name in data_names:
                    item_data.append(item[5][data_name])
                item_data = list(map(lambda x: 0.0 if x is None else x, item_data))

                if (label == 1):
                    goodBoardId.append(item[0])
                    goodBoardData.append(item_data)
                else:
                    badBoardId.append(item[0])
                    badBoardData.append(item_data)
        elif fqcflag == 1:
            for item in otherData:
                item_data = []
                for data_name in data_names:
                    item_data.append(item[5][data_name])
                item_data = list(map(lambda x: 0.0 if x is None else x, item_data))
                goodBoardId.append(item[0])
                goodBoardData.append(item_data)

        goodBoardDf = pd.DataFrame(data=goodBoardData, columns=data_names).fillna(0)
        goodBoardDf['upid'] = goodBoardId
        goodBoardData = np.array(goodBoardData)
        badBoardData = np.array(badBoardData)
        return goodBoardData, badBoardData, goodBoardId, badBoardId, labelArr, goodBoardDf
