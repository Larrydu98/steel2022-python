from ..models import GetDiagnosisDataDB
from ..models.GetDiagnosisDataDB import *
from ..utils import without_cooling_single_dimensional_variable, data_names_meas, diagnosisFlag, Cluster, single_dimensional_variable
import numpy as np
import pika, traceback
import pandas as pd
from scipy.stats import f
from scipy.stats import norm
import copy
import warnings


class DiaDataController():
    def __init__(self,start_time, end_time,minute_diff, merge_limit, merge_conflict,plate_limit,new_args,cooling):
        self.plate_limit = plate_limit
        # 获取分类数据
        cluster_data, cluster_column = GetDiagnosisDataDB.getAllDiaData(start_time,end_time)
        gantt_data = pd.DataFrame(data=cluster_data, columns=cluster_column).dropna(axis=0, how='all').reset_index(drop = True)
        if True:
            cluster = Cluster(gantt_data, minute_diff, merge_limit, merge_conflict, new_args, cooling)
            _, self.category_plate = cluster.getClusterData()

    def getDiaData(self):
        # res = []
        res = {}
        for bat_i, bat_val in enumerate(self.category_plate):
            # res.append({"batch_index": bat_i + 1, "category": []})
            for cat_i, cat_val in enumerate(bat_val):
                cate_detail, dia_res = {}, []
                if cat_val['merge_flag']:
                    test_df = cat_val['data'].loc[cat_val['data']['status_stats'] == 0]
                    train_rows, train_col_names = getTestTrainData(cat_val['category_info'], cat_val['data'].iloc[0, :]['platetype'], self.plate_limit)
                    if len(test_df) == 0 or len(train_rows) == 0:
                        cate_detail['category_index'] = cat_val["category_index"]
                        cate_detail['detail'] = dia_res
                        continue
                    else:
                        status_cooling = test_df.iloc[0,:]['status_cooling']
                        data_names = copy.deepcopy(single_dimensional_variable) if status_cooling == 0 else copy.deepcopy(without_cooling_single_dimensional_variable)
                        test_rows, test_cols = np.array(test_df).tolist(), test_df.columns.tolist()
                        try:
                            dia_class = createDiagResu()
                            cate_detail, _ = dia_class.run(train_rows, train_col_names, data_names, data_names_meas, test_rows, test_cols)

                        except Exception:
                            print(traceback.format_exc())

                else:
                    continue
                if cate_detail:
                    for i in cate_detail:
                        res[i['upid']] = i
                    # res.append(cate_detail)

        return res
# 获取测试集数据
def getTestTrainData(post_table,platetype, plate_limit):
    # re
    args = post_table
    label = ["tgtthickness", "tgtwidth", 'tgtlength', "tgtdischargetemp", "tgttmplatetemp", "cooling"]
    for i in label:
        if i in post_table:
            continue
        else:
            args[i] = []
    train_rows, train_col_names = GetDiagnosisDataDB.getDaignosisTrainData(args, platetype, plate_limit)
    return train_rows, train_col_names



class createDiagResu:
    '''
    createDiagResu
    '''

    def run(self, otherData, otherDataCol, data_names, data_names_meas, data, columns):
        '''
        run
        '''

        data, columns = data, columns
        # 这里的data是按照时间查找的要诊断钢板的数据
        # 诊断数据就是测试集划分的钢种
        data_df = pd.DataFrame(data=data, columns=columns).dropna(axis=0, how='any').reset_index(drop=True)
        if len(data_df) == 0:
            return [], 204

        labelArr = []
        goodBoardData = []
        goodBoardId = []
        for item in otherData:
            if item[otherDataCol.index('status_fqc')] == 0:
                if np.array(item[otherDataCol.index('fqc_label')]['method1']['data']).sum() == 5:
                    item_data = []
                    for data_name in data_names:
                        item_data.append(item[5][data_name])
                    item_data = list(map(lambda x: 0.0 if x is None else x, item_data))
                    goodBoardId.append(item[0])
                    goodBoardData.append(item_data)
                else:
                    continue

        goodBoardDf = pd.DataFrame(data=goodBoardData, columns=data_names).fillna(0)
        # print(len((goodBoardDf.max() - goodBoardDf.min())[(goodBoardDf.max() - goodBoardDf.min()) == 0]))
        goodBoardDf['upid'] = goodBoardId
        goodBoardData = np.array(goodBoardData)
        # badBoardData = np.array(badBoardData)

        X_train = np.array(goodBoardData)
        X_zero_std = np.where((np.std(X_train, axis=0)) <= 1e-10)
        X_train = np.delete(X_train, X_zero_std, axis=1)

        for i in sorted(X_zero_std[0], reverse=True):
            del data_names[i]

        if len(X_train) < 5:
            return [], 202

        train_len = len(goodBoardDf)
        diag_result = []
        for i in range(len(data)):
            upid = data[i][columns.index('upid')]
            # fqc_label = 1 if np.array(data[i][6]['method1']['data']).sum() == 5 else 0
            fqc_label = 1
            if data[i][columns.index('status_fqc')] == 1:
                fqc_label = 404
            else:
                if np.array(data[i][columns.index('label')]['method1']['data']).sum() == 5:
                    fqc_label = 1
                else:
                    fqc_label = 0
            toc = data[i][columns.index('toc')]
            one_process = []
            for data_name in data_names:
                if data[i][columns.index('stats')][data_name]:
                    one_process.append(data[i][columns.index('stats')][data_name])
                else:
                    one_process.append(0)
            one_process = list(map(lambda x: 0.0 if x is None else x, one_process))

            X_test = np.array(one_process)
            X_test =X_test.reshape((1, len(X_test)))
            # X_test = np.delete(X_test, X_zero_std, axis=1)

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

            diag_result.append({
                "upid": upid,
                "toc": toc.strftime("%Y-%m-%d %H:%M:%S"),
                "fqc_label": fqc_label,

                "one_dimens": result,
                "train_len": train_len,

                "CONTJ": CONTJ_Pro,
                "CONTQ": contq_Pro
            })
        return diag_result, 200

# 这里的都是算法
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # a = 1 / 0.0
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
    process_data = process_data.drop(columns=data_names_meas).values

    good_meas_data = good_data_df[data_names_meas]
    upid_meas_data = upid_data_df[data_names_meas]
    good_meas_data = good_meas_data[good_meas_data.sum(axis=1) > 0].values

    upid_process_data = upid_data_df.drop(columns=data_names_meas).values

    # plt.figure()
    # min_val = int(min(good_meas_data[:, 0])) + 1
    # max_val = int(max(good_meas_data[:, 0])) + 1
    # sns.displot(pd.Series(good_meas_data[:, 0]), bins=[i for i in range(min_val, max_val, 5)])
    # plt.grid()
    # plt.show()

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
    norm_process_data = ((process_data - process_data.min(axis=0)) / (process_data.max(axis=0) - process_data.min(axis=0)))#.fillna(0)
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
    norm_upid_data = ((upid_process_data - process_data.min(axis=0)) / (process_data.max(axis=0) - process_data.min(axis=0))).reshape(-1)
    norm_upid_data[np.isnan(norm_upid_data)] = 0
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
    meas_norm_process_data = ((meas_item_data - meas_item_data.min(axis=0)) / (meas_item_data.max(axis=0) - meas_item_data.min(axis=0)))#.fillna(0)
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
    meas_norm_upid_data = ((upid_meas_data - meas_item_data.min(axis=0)) / (meas_item_data.max(axis=0) - meas_item_data.min(axis=0))).values[0]

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
