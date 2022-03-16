import copy
from scipy.stats import f
from scipy.stats import norm
import numpy as np
import pika, traceback
import pandas as pd
import copy
from scipy.stats import f
from scipy.stats import norm
import numpy as np
import pika, traceback
import pandas as pd
from ..utils import single_dimensional_variable, without_cooling_single_dimensional_variable, data_names_meas, diagnosisFlag
from ..models.getDiagnosisDataDB import GetDiagnosisData
from  ..models.newgetDiagnosisDataDB import GetDiAllTestDataDB

class newdiagnosisDataComputer():
    def __init__(self, parser, start_time, end_time, merge_limit, merge_conflict, plate_limit):
        # 获取甘特图数据
        self.plate_limit = plate_limit
        label = ["cooling_rate1", "cooling_start_temp", "cooling_stop_temp", "fqcflag", "productcategory",
                 "slabthickness", "steelspec", "tgtdischargetemp", "tgtplatelength2",
                 "tgtplatethickness", "tgttmplatetemp", "tgtwidth", 'tgtthickness', 'tgtlength']
        for index in label:
            parser.add_argument(index, type=str, required=True)
        self.args = parser.parse_args(strict=True)
        post_table = {}
        post_table['tgtwidth'] = self.args['tgtwidth']
        post_table['tgtthickness'] = self.args['tgtthickness']
        post_table['tgtlength'] = self.args['tgtlength']
        gantt_chart = getGanttChartData(parser, start_time, end_time, merge_limit, merge_conflict, post_table)
        self.gantt_res = gantt_chart.getGanttChartData()

        self.diagnosisDataController()

    def diagnosisDataController(self):
        args = copy.deepcopy(self.args)
        new_args = {}
        for i in args:
            if args[i] == '[]':
                continue
            if i == 'tgtthickness' or i == 'tgtwidth' or i == 'tgtlength':
                continue
            new_args[i] = float(args[i])
        res = []
        for index, val in enumerate(self.gantt_res):
            res.append({
                "batch_index": val["batch"],
                "category": []
            })
            for category_index, category_val in enumerate(val['category']):
                cate_detail = {}
                res_args = {}
                # 获取数据
                for arg_i in new_args:
                    if arg_i == 'fqcflag':
                        res_args[arg_i] = new_args[arg_i]
                        continue
                    mid_test = new_args[arg_i]
                    res_args[arg_i] = [category_val['data'][arg_i].min() - mid_test, category_val['data'][arg_i].max() + mid_test]
                # res_args[arg_i] = str(category_val['data'].iloc[0]['status_cooling'])
                res_args['status_cooling'] = str(category_val['data'].iloc[0]['status_cooling'])
                res_args['upids'] = list(category_val['data']['upid'])
                # new_args['upids'] = list(category_val['data']['upid'])

                dia_class = diagnosisDataComputer(res_args, self.plate_limit)
                dia_res, status_code = dia_class.getdiagnosisData()
                cate_detail['category_index'] = category_val["category_index"]
                cate_detail['detail'] = dia_res
                res[index]['category'].append(cate_detail)
        return res






class diagnosisDataComputer():
    def __init__(self, args, plate_limit):
        # 获取数据
        get_data_class = GetDiagnosisData(args, plate_limit)
        self.train_rows, self.train_col_names, self.status_cooling, self.fqcflag = get_data_class.getDaignosisTrainData()
        self.train_df = pd.DataFrame(data=self.train_rows, columns=self.train_col_names)
        self.train_df['plate_quantity'] = diagnosisFlag(self.train_df, 'DataFrame')
        self.train_rows = self.train_df.values.tolist()
        self.test_rows, self.test_col_names = get_data_class.getDaignosisUpidData()
        self.single_dimensional_variable = copy.deepcopy(single_dimensional_variable) if self.status_cooling == 0 else copy.deepcopy(without_cooling_single_dimensional_variable)

    def getdiagnosisData(self):
        if len(self.train_rows) == 0 or len(self.test_rows) == 0:
            return [], 204
        try:
            _data_names = []
            if self.status_cooling == 0:
                _data_names = copy.deepcopy(single_dimensional_variable)
            elif self.status_cooling == 1:
                _data_names = copy.deepcopy(without_cooling_single_dimensional_variable)
            # 这是一种方法
            # diag_result, status_code = self.run(self.train_rows, _data_names, data_names_meas, self.fqcflag )


            diag_result, status_code = self.diagResu()
            return diag_result, status_code,
        except Exception:
            print(traceback.format_exc())
            return [], 500

    # 主程序入口



    def diagResu(self):
        '''

        '''
        # 这里的data是按照时间查找的要诊断钢板的数据
        good_board_data, good_board_id = [], []

        # fqcflag就是传进来的参数代表是否诊断特厚板 是1的话都考虑是0的话只考虑一种情况
        if self.fqcflag == 0:
            for item in self.train_rows:

                item_data = []
                # print(item[0])
                for data_name in self.single_dimensional_variable:
                    item_data.append(item[5][data_name])
                item_data = list(map(lambda x: 0.0 if x is None else x, item_data))
                if item[-1] == 0:
                    good_board_id.append(item[0])
                    good_board_data.append(item_data)
        elif self.fqcflag == 1:
            for item in self.train_rows:
                item_data = []
                for data_name in self.single_dimensional_variable:
                    item_data.append(item[5][data_name])
                item_data = list(map(lambda x: 0.0 if x is None else x, item_data))
                good_board_id.append(item[0])
                good_board_data.append(item_data)

        good_board_df = pd.DataFrame(data=good_board_data, columns=self.single_dimensional_variable).fillna(0)
        good_board_df['upid'] = good_board_id

        X_train = np.array(good_board_data)
        X_zero_std = np.where((np.std(X_train, axis=0)) <= 1e-10)
        X_train = np.delete(X_train, X_zero_std, axis=1)

        for i in sorted(X_zero_std[0], reverse=True):
            del self.single_dimensional_variable[i]

        if len(X_train) < 5:
            return [], 202

        test_data_df = pd.DataFrame(data=self.test_rows, columns=self.test_col_names)
        test_data_df['plate_quantity'] = diagnosisFlag(test_data_df, 'DataFrame')
        train_len = len(good_board_df)
        diag_result = []
        for index,val in test_data_df.iterrows():
            one_process = []
            for data_name in self.single_dimensional_variable:
                one_process.append(val['stats'][data_name])
            one_process = list(map(lambda x: 0.0 if x is None else x, one_process))
            X_test = np.array(one_process)
            X_test = X_test.reshape((1, len(X_test)))
            X_test = np.delete(X_test, X_zero_std, axis=1)
            T2UCL1, T2UCL2, QUCL, T2, Q, CONTJ, contq = self.general_call({
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
            upid_data_df = pd.DataFrame(data=X_test, columns=self.single_dimensional_variable)

            result = self.unidimensional_monitoring(upid_data_df,good_board_df, self.single_dimensional_variable,data_names_meas,0.25, 0.05, 0.01)
            diag_result.append({
                "upid": val['upid'],
                "toc": val['toc'].strftime("%Y-%m-%d %H:%M:%S"),
                "fqc_label": val['plate_quantity'],
                "one_dimens": result,
                "train_len": train_len,
                "CONTJ": CONTJ_Pro,
                "CONTQ": contq_Pro
            })


        return diag_result, 200

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
        # print(lamda[i, 0], i)
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

    def unidimensional_monitoring(self, upid_data_df, good_data_df, col_names, data_names_meas, quantile_num,
                                  extremum_quantile_num, s_extremum_quantile_num):
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
        upper_limit = np.quantile(process_data, 1 - quantile_num, axis=0)
        ## 计算原始过程数据的上下极值分位点
        extremum_lower_limit = np.quantile(process_data, extremum_quantile_num, axis=0)
        extremum_upper_limit = np.quantile(process_data, 1 - extremum_quantile_num, axis=0)
        ## 计算原始过程数据的上下超级极值分位点
        s_extremum_lower_limit = np.quantile(process_data, s_extremum_quantile_num, axis=0)
        s_extremum_upper_limit = np.quantile(process_data, 1 - s_extremum_quantile_num, axis=0)
        ## 查询此钢板的过程数据（若前端在点击马雷图或散点图前已经储存该数据，则此步骤可以省略）
        upid_data = upid_data_df.values[0]
        ## 对同一规格钢板过程数据进行归一化计算
        norm_process_data = ((process_data - process_data.min(axis=0)) / (
                    process_data.max(axis=0) - process_data.min(axis=0)))  # .fillna(0)
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
        norm_upid_data = ((upid_process_data - process_data.min(axis=0)) / (
                    process_data.max(axis=0) - process_data.min(axis=0))).reshape(-1)
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
        meas_norm_process_data = ((meas_item_data - meas_item_data.min(axis=0)) / (
                    meas_item_data.max(axis=0) - meas_item_data.min(axis=0)))  # .fillna(0)
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
        meas_norm_upid_data = ((upid_meas_data - meas_item_data.min(axis=0)) / (
                    meas_item_data.max(axis=0) - meas_item_data.min(axis=0))).values[0]

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



# 获取甘特图的数据
class getGanttChartData():
    def __init__(self, parser, start_time, end_time,merge_limit,merge_conflict,post_table):
        rows, col_names = GetDiAllTestDataDB.getDiaTestDataDB(start_time, end_time)
        self.post_table = post_table
        self.gantt_data = pd.DataFrame(data=rows, columns=col_names).dropna(axis=0, how='all').reset_index(drop=True)
        self.merge_limit = int(merge_limit)
        self.merge_conflict = int(merge_conflict)

    def getGanttChartData(self):
        # 批次划分的时间差
        minute_diff = 30
        result = []
        batch_plate = self.batchPlate(minute_diff)
        # 是否通过原始聚类方法
        if True:
            category_plate = self.categoryPlate(batch_plate)
            for bat_index,bat_vla in enumerate(category_plate):
                result.append({
                    'batch': bat_index + 1,
                    'category': []
                })
                result[bat_index]['batch'] = bat_index + 1
                for cat_index,cat_val in enumerate(bat_vla):
                    if cat_val['merge_flag']:
                        result[bat_index]['category'].append({
                            'category_index': cat_index+1,
                            "data": cat_val['data']
                        })



        # for index, value in enumerate(category_plate):
        #     good_flag, bad_flag, no_flag = CountGoodBadNoflag(batch_plate[index])
        #     result.append({
        #         'batch': index + 1,
        #         'good_flag': good_flag,
        #         'bad_flag': bad_flag,
        #         'no_flag': no_flag,
        #         'total': len(batch_plate[index]),
        #         'startTime': batch_plate[index].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
        #         'endTime': batch_plate[index].iloc[len(batch_plate[index]) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
        #         "tgtwidth_avg": round(batch_plate[index].tgtwidth.mean(), 3),
        #         "tgtlength_avg": round(batch_plate[index].tgtlength.mean(), 3),
        #         "tgtthickness_avg": round(batch_plate[index].tgtthickness.mean(), 5),
        #         'category': []
        #     })
        #     # print('====================================')
        #     for i, val in enumerate(value):
        #         good_flag, bad_flag, no_flag = CountGoodBadNoflag(val['data'])
        #         # platetype = val.iloc[0,:]['platetype']
        #         if val['merge_flag']:
        #             result[index]['category'].append({
        #                 'merge_flag': val['merge_flag'],
        #                 'category_index': val['category_index'],
        #                 'category_info': val['category_info'],
        #                 'platetype': val['data'].iloc[0, :]['platetype'],
        #                 'good_flag': good_flag,
        #                 'bad_flag': bad_flag,
        #                 'no_flag': no_flag,
        #                 'total': len(val['data']),
        #                 'startTime': val['data'].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
        #                 'endTime': val['data'].iloc[len(val['data']) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
        #                 "tgtwidth_avg": round(val['data'].tgtwidth.mean(), 3),
        #                 "tgtlength_avg": round(val['data'].tgtlength.mean(), 3),
        #                 "tgtthickness_avg": round(val['data'].tgtthickness.mean(), 5),
        #                 "tgtthickness_avg": round(val['data'].tgtthickness.mean(), 5),
        #                 'detail': []
        #
        #             })
        #         else:
        #             result[index]['category'].append({
        #                 'merge_flag': val['merge_flag'],
        #                 'category_index': val['category_index'],
        #                 'platetype': val['data'].iloc[0, :]['platetype'],
        #                 'good_flag': good_flag,
        #                 'bad_flag': bad_flag,
        #                 'no_flag': no_flag,
        #                 'total': len(val['data']),
        #                 'startTime': val['data'].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
        #                 'endTime': val['data'].iloc[len(val['data']) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
        #                 "tgtwidth_avg": round(val['data'].tgtwidth.mean(), 3),
        #                 "tgtlength_avg": round(val['data'].tgtlength.mean(), 3),
        #                 "tgtthickness_avg": round(val['data'].tgtthickness.mean(), 5),
        #                 "tgtthickness_avg": round(val['data'].tgtthickness.mean(), 5),
        #                 'detail': []
        #             })
        #         # print(len(val['data']), val['merge_flag'])
        #         # print(val['platetype'])
        #
        #         for df_index, df_row in val['data'].iterrows():
        #             result[index]['category'][i]['detail'].append({
        #                 'upid': df_row['upid'],
        #                 'toc': df_row['toc'].strftime("%Y-%m-%d %H:%M:%S"),
        #                 'platetype': df_row['platetype'],
        #                 'tgtwidth': df_row['tgtwidth'],
        #                 'tgtlength': df_row['tgtlength'],
        #                 'tgtthickness': df_row['tgtthickness'],
        #                 'flag_lable': self.JudgePlateQuality(df_row),
        #                 'status_cooling': df_row['status_cooling']
        #             })

        # #
        return result

        # 批次类别

    def batchPlate(self, minute_diff):
        batch_plate = []
        minute_diff = minute_diff - 1

        time_series = self.gantt_data['toc'].diff().astype('timedelta64[m]').loc[
            self.gantt_data['toc'].diff().astype('timedelta64[m]') > minute_diff]
        index_array = time_series.index.tolist()
        if len(index_array) == 0:
            batch_plate.append(self.gantt_data)
        else:
            for index, value in enumerate(index_array):
                if index == 0:
                    batch_plate.append(self.gantt_data.iloc[0:value, ])
                elif index == len(index_array) - 1:
                    batch_plate.append(self.gantt_data.iloc[value:, ])
                else:
                    batch_plate.append(self.gantt_data.iloc[value:index_array[index + 1], ])

        return batch_plate

        # 划分种类并判断是否可以合并

    def categoryPlate(self, batch_plate):
        # 判断合并冲突的最大极限
        merrge_confict = 4
        res = []
        for batch_index, batch_val in enumerate(batch_plate):
            # 1是不能合并，0是可以合并
            category_plate = []
            might_merge_index = batch_val['platetype'].value_counts()[
                batch_val['platetype'].value_counts() >= self.merge_limit].index.tolist()
            cannot_merge_index = batch_val['platetype'].value_counts()[
                batch_val['platetype'].value_counts() < self.merge_limit].index.tolist()

            for index, value in enumerate(cannot_merge_index):
                category_plate.append({'merge_flag': False, 'data': batch_val.loc[batch_val['platetype'] == value]})
            for i, val in enumerate(might_merge_index):
                might_merge_arr_index = batch_val.loc[batch_val['platetype'] == val].index
                # 新增长度宽度厚度三个选择
                might_merge_df = batch_val.loc[batch_val['platetype'] == val]
                coding_list, specification_list = self.DataframeLable(might_merge_df)
                # 编码
                might_merge_df.insert(loc=len(might_merge_df.columns), column='coding', value=coding_list)
                groupby_df = might_merge_df.groupby(might_merge_df['coding']).count()
                unable_merge_list = list(groupby_df.drop(groupby_df[groupby_df.upid >= self.merge_limit].index).index)
                might_able_merge_list = list(
                    groupby_df.drop(groupby_df[groupby_df.upid < self.merge_limit].index).index)
                for unable_merge_i in unable_merge_list:
                    category_plate.append({'merge_flag': False, 'data': batch_val.loc[
                        might_merge_df[might_merge_df['coding'] == unable_merge_i].index]})

                for might_able_merge_list_i in might_able_merge_list:
                    result = self.JudgeMerge(
                        might_merge_df[might_merge_df['coding'] == might_able_merge_list_i].index.values.tolist())
                    for res_index, res_val in enumerate(result):
                        # print(i, res_index)
                        if len(res_val) >= self.merge_limit:
                            category_info = {}
                            for spe_i, spe_vla in enumerate(specification_list):
                                category_info[spe_vla] = specification_list[spe_vla][
                                    int(might_merge_df.loc[res_val].iloc[0, :]['coding'][spe_i])]
                            category_info['cooling'] = int(might_merge_df.loc[res_val].iloc[0, :]['coding'][-1])
                            category_plate.append(
                                {'merge_flag': True, 'data': batch_val.loc[res_val], "category_info": category_info})
                        else:
                            category_plate.append({'merge_flag': False, 'data': batch_val.loc[res_val]})
            category_plate.sort(key=lambda k: (k.get('data').iloc[0].toc))
            for category_plate_index, category_plate_val in enumerate(category_plate):
                category_plate_val['category_index'] = category_plate_index + 1
            res.append(category_plate)
        return res

        # 判断单块钢板的好坏以及404

    def JudgePlateQuality(self, row):
        if (row['status_fqc'] == 0) & (row['flag_lable'] == '[1, 1, 1, 1, 1]'):
            return 0
        elif (row['status_fqc'] == 0) & (row['flag_lable'] != '[1, 1, 1, 1, 1]'):
            return 1
        else:
            return 404

    def JudgeMerge(self, data):
        interval_list = []
        index_location = 0
        for index, val in enumerate(data):
            if index == 0:
                continue
            else:
                if val - data[index - 1] > self.merge_conflict:
                    interval_list.append(data[index_location: index])
                    index_location = index
        interval_list.append(data[index_location:])
        return interval_list

        # 对dataframe进行打标签

    def DataframeLable(self, data):
        # 厚度0.01宽度宽度0.8，长度4
        coding_list = [''] * len(data)
        specification_list = {}
        for val in self.post_table:
            specification_list[val] = []
            label_max = data[val].max()
            label_min = data[val].min()
            s_bin = []
            point_move = label_min
            while point_move <= label_max:
                s_bin.append(point_move)
                point_move += float(self.post_table[val])
            s_bin.append(point_move)

            for i in range(len(s_bin)):
                if i < len(s_bin) - 1:
                    specification_list[val].append([s_bin[i], s_bin[i + 1]])
            for index, pd_val in enumerate(pd.cut(data[val], s_bin, labels=False, right=False)):
                coding_list[index] += str(pd_val)

        for coding_index in range(len(coding_list)):
            coding_list[coding_index] += str(data.iloc[coding_index, :]['status_cooling'])

        return coding_list, specification_list

