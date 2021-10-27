'''
createDiagResu
'''
# https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import f
from scipy.stats import norm
from ...utils import new_getData
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

def unidimensional_monitoring(upid_data_df, good_data_df, col_names, data_names_meas, quantile_num, extremum_quantile_num):

    ## 获取同一规格钢板的取过程数据
    process_data = good_data_df[col_names]
    meas_data = good_data_df[data_names_meas]
    upid_meas_data = upid_data_df[data_names_meas]
    # meas_data_after_drop = good_data_df.drop(good_data_df[good_data_df[data_names_meas[0]] == 0].index)
    ## 计算原始过程数据的上下分位点
    lower_limit  = process_data.quantile(q = quantile_num, axis = 0).values
    upper_limit  = process_data.quantile(q = 1-quantile_num, axis = 0).values
    ## 计算原始过程数据的上下极值分位点
    extremum_lower_limit  = process_data.quantile(q = extremum_quantile_num, axis = 0).values
    extremum_upper_limit  = process_data.quantile(q = 1-extremum_quantile_num, axis = 0).values
    ## 查询此钢板的过程数据（若前端在点击马雷图或散点图前已经储存该数据，则此步骤可以省略）
    upid_data = upid_data_df.values[0]
    ## 对同一规格钢板过程数据进行归一化计算
    norm_process_data = ((process_data - process_data.min()) / (process_data.max() - process_data.min())).fillna(0)
    ## 计算归一化后过程数据的上下分位点
    lower_limit_norm = norm_process_data.quantile(q = quantile_num, axis = 0).values
    upper_limit_norm = norm_process_data.quantile(q = 1-quantile_num, axis = 0).values
    ## 计算归一化后过程数据的上下极值分位点
    extremum_lower_limit_norm = norm_process_data.quantile(q = extremum_quantile_num, axis = 0).values
    extremum_upper_limit_norm = norm_process_data.quantile(q = 1-extremum_quantile_num, axis = 0).values
    ## 查询此钢板归一化后的过程数据
    # norm_upid_data = norm_process_data[good_data_df.upid == upid][col_names].values[0]
    norm_upid_data = ((upid_data_df - process_data.min()) / (process_data.max() - process_data.min())).fillna(0).values[0]
    norm_upid_data[norm_upid_data > 1] = 1
    norm_upid_data[norm_upid_data < 0] = 0
    ## 计算归一化后超限幅度
    over_limit_range = copy.copy(norm_upid_data)
    extremum_over_limit_range = copy.copy(norm_upid_data)

    over_limit_range[np.where(norm_upid_data > upper_limit_norm)] = norm_upid_data[norm_upid_data > upper_limit_norm] - upper_limit_norm[norm_upid_data > upper_limit_norm]
    over_limit_range[np.where(norm_upid_data < lower_limit_norm)] = norm_upid_data[norm_upid_data < lower_limit_norm] - lower_limit_norm[norm_upid_data < lower_limit_norm]
    over_limit_range[np.where((norm_upid_data >= lower_limit_norm) & (norm_upid_data <= upper_limit_norm))] = 0

    extremum_over_limit_range[np.where(norm_upid_data > extremum_upper_limit_norm)] = norm_upid_data[norm_upid_data > extremum_upper_limit_norm] - extremum_upper_limit_norm[norm_upid_data > extremum_upper_limit_norm]
    extremum_over_limit_range[np.where(norm_upid_data < extremum_lower_limit_norm)] = norm_upid_data[norm_upid_data < extremum_lower_limit_norm] - extremum_lower_limit_norm[norm_upid_data < extremum_lower_limit_norm]
    extremum_over_limit_range[np.where((norm_upid_data >= extremum_lower_limit_norm) & (norm_upid_data <= extremum_upper_limit_norm))] = 0
    result = []
    for i in range(len(col_names)):
        # print(col_names[i])
        if col_names[i] in data_names_meas:
            meas_item_data = meas_data[col_names[i]]
            meas_item_data = meas_item_data[meas_item_data != 0]

            ## 计算原始过程数据的上下分位点
            meas_lower_limit = meas_item_data.quantile(q=quantile_num)
            meas_upper_limit = meas_item_data.quantile(q=1 - quantile_num)
            ## 计算原始过程数据的上下极值分位点
            meas_extremum_lower_limit = meas_item_data.quantile(q=extremum_quantile_num)
            meas_extremum_upper_limit = meas_item_data.quantile(q=1 - extremum_quantile_num)
            ## 查询此钢板的过程数据（若前端在点击马雷图或散点图前已经储存该数据，则此步骤可以省略）
            upid_data = upid_data_df.values[0]
            ## 对同一规格钢板过程数据进行归一化计算
            meas_norm_process_data = ((meas_item_data - meas_item_data.min()) / (meas_item_data.max() - meas_item_data.min())).fillna(0)
            ## 计算归一化后过程数据的上下分位点
            meas_lower_limit_norm = meas_norm_process_data.quantile(q=quantile_num)
            meas_upper_limit_norm = meas_norm_process_data.quantile(q=1 - quantile_num)
            ## 计算归一化后过程数据的上下极值分位点
            meas_extremum_lower_limit_norm = meas_norm_process_data.quantile(q=extremum_quantile_num)
            meas_extremum_upper_limit_norm = meas_norm_process_data.quantile(q=1 - extremum_quantile_num)
            ## 查询此钢板归一化后的过程数据
            meas_norm_upid_data = ((upid_meas_data[col_names[i]] - meas_item_data.min()) / (meas_item_data.max() - meas_item_data.min())).values[0]
            if meas_norm_upid_data < 0:
                meas_norm_upid_data = None
            # ## 计算归一化后超限幅度
            # over_limit_range = copy.copy(norm_upid_data)
            # extremum_over_limit_range = copy.copy(norm_upid_data)


            # extremum_lower_limit_norm[i] = ?
            # extremum_upper_limit_norm[i] = ?
            # extremum_lower_limit[i] =
            # extremum_upper_limit[i] =
            result.append({
                'name': col_names[i],
                'value': meas_norm_upid_data,
                'l': meas_lower_limit_norm,
                'u': meas_upper_limit_norm,
                'extremum_l': meas_extremum_lower_limit_norm,
                'extremum_u': meas_extremum_upper_limit_norm,
                'original_value': upid_data[i],
                'original_l': meas_lower_limit,
                'original_u': meas_upper_limit,
                'extremum_original_l': meas_extremum_lower_limit,
                'extremum_original_u': meas_extremum_upper_limit
            })
        else:
            result.append({
                'name': col_names[i],
                'value': norm_upid_data[i],
                'l': lower_limit_norm[i],
                'u': upper_limit_norm[i],
                'extremum_l': extremum_lower_limit_norm[i],
                'extremum_u': extremum_upper_limit_norm[i],
                'original_value': upid_data[i],
                'original_l': lower_limit[i],
                'original_u': upper_limit[i],
                'extremum_original_l': extremum_lower_limit[i],
                'extremum_original_u': extremum_upper_limit[i]
                })
    return result, over_limit_range, extremum_over_limit_range

class createDiagResu:
    '''
    createDiagResu
    '''

    def __init__(self, upid):
        self.upid = upid
        self.paramsIllegal = False
        try:
            # 如果以后加入参数，在这里加入
            # print('createDiagResu方法暂时没有参数')
            pass
        except Exception:
            self.paramsIllegal = True

    def run(self, otherData, data_names, data_names_meas, sorttype, fqcflag):
        '''
        run
        '''
        # print(self.upid)
        ismissing = {'status_stats':True}
        data,columns = new_getData(['upid', 'platetype', 'tgtwidth','tgtlength','tgtthickness','stats','fqc_label', 'toc'], ismissing, [], [], [], [], [self.upid], [], '', '')
        data_df = pd.DataFrame(data=data, columns=columns).dropna(axis=0, how='any').reset_index(drop=True)
        if len(data_df) == 0:
            return 400, 400, 400, 400

        process_data = []
        for data_name in data_names:
            process_data.append(data[0][5][data_name])
        process_data = list(map(lambda x: 0.0 if x is None else x, process_data))

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


        goodBoardDf = pd.DataFrame(data = goodBoardData, columns = data_names).fillna(0)
        # print(len((goodBoardDf.max() - goodBoardDf.min())[(goodBoardDf.max() - goodBoardDf.min()) == 0]))
        goodBoardDf['upid'] = goodBoardId
        goodBoardData = np.array(goodBoardData)
        # badBoardData = np.array(badBoardData)
        # X_test = np.array(badBoardData)
        X_test = np.array(process_data)
        X_test = X_test.reshape((1, len(X_test)))
        X_train = np.array(goodBoardData)

        # X_train = np.nan_to_num(X_train, nan=0)

        X_zero_std = np.where((np.std(X_train, axis=0)) <= 1e-10)
        X_train = np.delete(X_train, X_zero_std, axis=1)
        X_test = np.delete(X_test, X_zero_std, axis=1)

        if len(X_train) < 5:
            return {}, {}, {}, {}, 0

        T2UCL1, T2UCL2, QUCL, T2, Q, CONTJ, contq = PCATEST().general_call({
            'Xtrain': X_train,
            'Xtest': X_test,
            })

        for i in sorted(X_zero_std[0], reverse=True):
            del data_names[i]

        upid_data_df = pd.DataFrame(data=X_test, columns=data_names)
        result, over_limit_range, extremum_over_limit_range = unidimensional_monitoring(upid_data_df, goodBoardDf, data_names, data_names_meas, 0.25, 0.05)
        CONTJ_Pro = []
        maxCON = max(CONTJ)
        minCON = min(CONTJ)
        for item in CONTJ:
            mid = (item - minCON)/(maxCON - minCON)
            CONTJ_Pro.append(mid)

        contq_Pro = []
        maxContq = max(contq.tolist())
        minContq = min(contq.tolist())
        for item in contq.tolist():
            mid = (item - minContq)/(maxContq - minContq)
            contq_Pro.append(mid)


        if sorttype != "default":
            sort_result = pd.DataFrame({
                "data_names": data_names,
                "result": result,
                "over_limit_range": over_limit_range,
                "extremum_over_limit_range": extremum_over_limit_range,
                "CONTJ_Pro": CONTJ_Pro,
                "contq_Pro": contq_Pro
            })
            sort_result["extremum_over_limit_range_abs"] = sort_result.extremum_over_limit_range.abs()
            sort_result["over_limit_range_abs"] = sort_result.over_limit_range.abs()
            if sorttype == "uni":
                sort_result.sort_values(by=['extremum_over_limit_range_abs', 'over_limit_range_abs', 'contq_Pro', 'CONTJ_Pro'],
                                        ascending=[False, False, False, False],
                                        inplace=True)

            elif sorttype == "t2":
                sort_result.sort_values(by=['CONTJ_Pro', 'contq_Pro', 'extremum_over_limit_range_abs', 'over_limit_range_abs'],
                                        ascending=[False, False, False, False],
                                        inplace=True)
            elif sorttype == "spe":
                sort_result.sort_values(by=['contq_Pro', 'CONTJ_Pro', 'extremum_over_limit_range_abs', 'over_limit_range_abs'],
                                        ascending=[False, False, False, False],
                                        inplace=True)

            data_names = sort_result.data_names.values.tolist()
            result = sort_result.result.values.tolist()
            over_limit_range = sort_result.over_limit_range.values
            extremum_over_limit_range = sort_result.extremum_over_limit_range.values
            CONTJ_Pro = sort_result.CONTJ_Pro.values.tolist()
            contq_Pro = sort_result.contq_Pro.values.tolist()


        outOfGau = {
            'xData': data_names,
            'over_limit_range': over_limit_range.tolist(),
            'extremum_over_limit_range': extremum_over_limit_range.tolist()
        }
        PCAT2 = {
            'xData': data_names,
            'sData': CONTJ_Pro
        }
        PCASPE = {
            'xData': data_names,
            'sData': contq_Pro
        }
        # result = unidimensional_monitoring(self.upid, goodBoardDf, 0.25, 0.05, data_names)


        return result, outOfGau, PCAT2, PCASPE, len(goodBoardDf)
