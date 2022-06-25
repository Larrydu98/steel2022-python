from ..models.VisualizationTsneDB import *
from ..utils import single_dimensional_variable,without_cooling_single_dimensional_variable,Cluster, countGoodBadNoflag
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import datetime as dt
specifications = ["tgtwidth", "tgtplatelength2", "productcategory", "steelspec"]

class VisualizationTsneController:
    def __init__(self, start_time, end_time,minute_diff, merge_limit, merge_conflict, args, cooling):
        self.plate_yield_data, self.column_names = VisualizationTsneDB.getPlateYieldStaisticsData(start_time, end_time)
        self.tsne_data = pd.DataFrame(data=self.plate_yield_data, columns=self.column_names).dropna(axis=0, how='all').reset_index(drop=True)
        if True:
            cluster = Cluster(self.tsne_data, minute_diff, merge_limit, merge_conflict, args, cooling)
            _, self.category_plate = cluster.getClusterData()

    def visualizationTsenController(self):
        production_rhythm_list = []
        tsne_data = []
        upids_arr = []
        all_upid = np.array([])
        for cat_i, cat_val in enumerate(self.category_plate):
            for i, val in enumerate(cat_val):
                if val['merge_flag']:
                    all_upid = np.append(all_upid, val['data']['upid'].values)
                    upids_arr.append(val['data']['upid'].values.tolist())
                    tsne_data.append({"bat_index": cat_i + 1,
                                      "cat_index": i + 1,
                                      "data": val['data']})
        X = []
        for df_index, df_val in enumerate(tsne_data):
            all_process = []
            for i, val in df_val['data'].iterrows():
                process_data = []
                if val['status_cooling'] == 0:
                    for data_name in single_dimensional_variable:
                        process_data.append(val['stats'][data_name])
                elif val['status_cooling'] == 1:
                    for data_name in without_cooling_single_dimensional_variable:
                        process_data.append(val['stats'][data_name])
                all_process.append(process_data)
            all_process = pd.DataFrame(all_process).fillna(0).mean().values.tolist()
            X.append(all_process)
        X = pd.DataFrame(X).fillna(0).values.tolist()
        X_embedded = TSNE(n_components=2).fit_transform(X)
        all_upid = str(all_upid.tolist()).replace('[', '(').replace(']', ')')
        info_row, info_column = VisualizationTsneDB.getCurrentData(all_upid)
        info_df = pd.DataFrame(data=info_row, columns=info_column).dropna(axis=0, how='all').reset_index(drop=True)

        res = [None] * len((X_embedded))
        for i in range(len(X_embedded)):
            good, bad, no = countGoodBadNoflag(tsne_data[i]['data'], 'dataframe')
            # print(i)
            # if i == 18:
            #     print('debug')
            res[i] = {}
            merge_df = info_df[info_df['upid'].isin(upids_arr[i])].dropna(axis=0,how='any')
            if len(merge_df) == 0:
                continue
            process_df = self.processDataframe(merge_df)
            production_rhythm = (merge_df.iloc[-1]['toc'] - merge_df.iloc[0]['toc']).total_seconds() / len(merge_df)
            production_rhythm_list.append(production_rhythm)
            # max_list.append(process_df.max())
            # process_list.append(process_df)
            process_mean = process_df.mean()
            res[i]['bat_index'] = tsne_data[i]['bat_index']
            res[i]['cat_index'] = tsne_data[i]['cat_index']
            res[i]['good'] = good
            res[i]['bad'] = bad
            res[i]['no'] = no


            res[i]['x'] = float(X_embedded[i][0])
            res[i]['y'] = float(X_embedded[i][1])
            res[i]['details'] = {
                'production_rhythm': production_rhythm,
                'heating_mean': [process_mean['heat1'], process_mean['heat2'],
                                                                  process_mean['heat3'], process_mean['heat4'],
                                                                  process_mean['heat5']],
                'rolling_mean': [process_mean['RmF3Pass'], process_mean['RmL3Pass'], process_mean['RmEnd'], process_mean['FmStart'], process_mean['FmF3Pass'],process_mean['FmL3Pass'], process_mean['FmEnd']],
                'cooling_mean': [process_mean['CcDQEnd'], process_mean['CcACCEnd']],
                'total_mean': [process_mean['heat_total'], process_mean['rolling_total'], process_mean['CcTotal']],
                'total_var':[process_df['heat_total'].var(), process_df['rolling_total'].var(), process_df['CcTotal'].var()]
            }
            # res[i]['production_rhythm'] = production_rhythm
            # res[i]['heating_mean'] = [process_mean['heat1'], process_mean['heat2'],
            #                                                       process_mean['heat3'], process_mean['heat4'],
            #                                                       process_mean['heat5']]
            # res[i]['rolling_mean'] = [process_mean['RmF3Pass'], process_mean['RmL3Pass'],
            #                                                       process_mean['RmEnd'], process_mean['FmStart'],
            #                                                       process_mean['FmF3Pass'],
            #                                                       process_mean['FmL3Pass'], process_mean['FmEnd']]
            # res[i]['cooling_mean'] = [process_mean['CcDQEnd'], process_mean['CcACCEnd']]
            # res[i]['total_mean'] = [process_mean['heat_total'],
            #                                                     process_mean['rolling_total'], process_mean['CcTotal']]
            # res[i]['total_var'] = [process_df['heat_total'].var(),
            #                                                    process_df['rolling_total'].var(),
            #                                                    process_df['CcTotal'].var()]

            # res['series' + str(i + 1)]['production_rhythm'] = production_rhythm
            # res['series' + str(i + 1)]['heating_mean'] = [process_mean['heat1'], process_mean['heat2'],
            #                                                       process_mean['heat3'], process_mean['heat4'],
            #                                                       process_mean['heat5']]
            # res['series' + str(i + 1)]['rolling_mean'] = [process_mean['RmF3Pass'], process_mean['RmL3Pass'],
            #                                                       process_mean['RmEnd'], process_mean['FmStart'],
            #                                                       process_mean['FmF3Pass'],
            #                                                       process_mean['FmL3Pass'], process_mean['FmEnd']]
            # res['series' + str(i + 1)]['cooling_mean'] = [process_mean['CcDQEnd'], process_mean['CcACCEnd']]
            # res['series' + str(i + 1)]['total_mean'] = [process_mean['heat_total'],
            #                                                     process_mean['rolling_total'], process_mean['CcTotal']]
            # res['series' + str(i + 1)]['total_var'] = [process_df['heat_total'].var(),
            #                                                    process_df['rolling_total'].var(),
            #                                                    process_df['CcTotal'].var()]
        return res

    # 获取种类中的加热轧制冷却的时间占比信息
    def processDataframe(self, data):
        heat_col_names = ["heat1", "heat2", "heat3", "heat4", "heat5", "heat_total"]
        heat_rows = []
        rolling_cols = ['RmF3Pass', 'RmL3Pass', 'RmEnd', 'FmStart', 'FmF3Pass', 'FmL3Pass', 'FmEnd', 'rolling_total']
        rolling_rows = []
        cooling_cols = ['CcDQEnd', 'CcACCEnd', 'CcTotal']
        cooling_rows = []
        for plate_index, plate_val in data.iterrows():
            fu_discharge_time = plate_val.discharge_time
            # 0101
            # 加热
            heat1 = dt.timedelta(minutes=plate_val.staying_time_pre)
            heat2 = dt.timedelta(minutes=plate_val.staying_time_1)
            heat3 = dt.timedelta(minutes=plate_val.staying_time_2)
            heat4 = dt.timedelta(minutes=plate_val.staying_time_soak)
            heat5 = (fu_discharge_time - (
                    fu_discharge_time - dt.timedelta(minutes=plate_val.in_fce_time) + heat1 + heat2 + heat3 + heat4))
            # 总时间
            heat_total = dt.timedelta(minutes=plate_val.in_fce_time)
            heat_data = [heat1.total_seconds(), heat2.total_seconds(), heat3.total_seconds(), heat4.total_seconds(),
                         heat5.total_seconds(), heat_total.total_seconds()]
            heat_rows.append(heat_data)
            # 轧制
            rm_list, fm_list = [], []
            rolling_dic = {}
            cooling_list = []
            cooling_row = {}
            for stop_index, stop_val in enumerate(plate_val["stops"]):
                if 'RM' in stop_val['station']['name']:
                    rm_list.append({'index': stop_index, 'name': stop_val['station']['name'],
                                    'real_time': dt.datetime.strptime(stop_val['realTime'], '%Y-%m-%d %H:%M:%S'),
                                    'time': stop_val['time']})
                elif 'FM' in stop_val['station']['name']:
                    fm_list.append({'index': stop_index, 'name': stop_val['station']['name'],
                                    'real_time': dt.datetime.strptime(stop_val['realTime'], '%Y-%m-%d %H:%M:%S'),
                                    'time': stop_val['time']})
                elif 'Cc' in stop_val['station']['name']:
                    cooling_list.append({'index': stop_index, 'name': stop_val['station']['name'],
                                         'real_time': dt.datetime.strptime(stop_val['realTime'], '%Y-%m-%d %H:%M:%S'),
                                         'time': stop_val['time']})

            # 粗轧
            if len(fm_list) == 0:
                rolling_dic["FmStart"] = 0
                rolling_dic["FmF3Pass"] = 0
                rolling_dic["FmL3Pass"] = 0
                rolling_dic["FmEnd"] = 0
            elif len(rm_list) <= 3 and len(rm_list) > 0:
                RmF3Pass = rm_list[-1]['real_time'] - rm_list[0]['real_time']

                rolling_dic["RmF3Pass"] = RmF3Pass.total_seconds()
                rolling_dic["RmL3Pass"] = 0
                rolling_dic["RmEnd"] = 0
            elif len(rm_list) > 3 and len(rm_list) <= 6:
                RmF3Pass = rm_list[2]['real_time'] - rm_list[0]['real_time']
                RmL3Pass = rm_list[-1]['real_time'] - rm_list[2]['real_time']

                rolling_dic["RmF3Pass"] = RmF3Pass.total_seconds()
                rolling_dic["RmL3Pass"] = RmL3Pass.total_seconds()
                rolling_dic["RmEnd"] = 0
            elif len(rm_list) > 6:
                RmF3Pass = rm_list[2]['real_time'] - rm_list[0]['real_time']
                RmL3Pass = rm_list[-3]['real_time'] - rm_list[2]['real_time']
                RmEnd = rm_list[-1]['real_time'] - rm_list[-3]['real_time']

                rolling_dic["RmF3Pass"] = RmF3Pass.total_seconds()
                rolling_dic["RmL3Pass"] = RmL3Pass.total_seconds()
                rolling_dic["RmEnd"] = RmEnd.total_seconds()
            # 精轧
            if len(fm_list) == 0:
                rolling_dic["FmStart"] = 0
                rolling_dic["FmF3Pass"] = 0
                rolling_dic["FmL3Pass"] = 0
                rolling_dic["FmEnd"] = 0
            elif len(fm_list) <= 3 and len(fm_list) > 0:
                FmStart = fm_list[0]['real_time'] - rm_list[-1]['real_time']
                FmF3Pass = fm_list[-1]['real_time'] - rm_list[0]['real_time']

                rolling_dic["FmStart"] = FmStart.total_seconds()
                rolling_dic["FmF3Pass"] = FmF3Pass.total_seconds()
                rolling_dic["FmL3Pass"] = 0
                rolling_dic["FmEnd"] = 0
            elif len(fm_list) > 3 and len(fm_list) <= 6:
                FmStart = fm_list[0]['real_time'] - rm_list[-1]['real_time']
                FmF3Pass = fm_list[2]['real_time'] - fm_list[0]['real_time']
                FmL3Pass = fm_list[-1]['real_time'] - fm_list[2]['real_time']

                rolling_dic["FmStart"] = FmStart.total_seconds()
                rolling_dic["FmF3Pass"] = FmF3Pass.total_seconds()
                rolling_dic["FmL3Pass"] = FmL3Pass.total_seconds()
                rolling_dic["FmEnd"] = 0
            elif len(fm_list) > 6:
                FmStart = fm_list[0]['real_time'] - rm_list[-1]['real_time']
                FmF3Pass = fm_list[2]['real_time'] - fm_list[0]['real_time']
                FmL3Pass = fm_list[-3]['real_time'] - fm_list[2]['real_time']
                FmEnd = fm_list[-1]['real_time'] - fm_list[-3]['real_time']

                rolling_dic["FmStart"] = FmStart.total_seconds()
                rolling_dic["FmF3Pass"] = FmF3Pass.total_seconds()
                rolling_dic["FmL3Pass"] = FmL3Pass.total_seconds()
                rolling_dic["FmEnd"] = FmEnd.total_seconds()
            # 冷却
            if len(cooling_list) != 0:
                if len(cooling_list) == 2:
                    cooling_row["CcDQEnd"] = (
                                cooling_list[-1]['real_time'] - cooling_list[0]['real_time']).total_seconds()
                    cooling_row["CcACCEnd"] = 0
                    cooling_row["CcTotal"] = (
                                cooling_list[-1]['real_time'] - cooling_list[0]['real_time']).total_seconds()
                else:
                    cooling_row["CcDQEnd"] = (
                                cooling_list[1]['real_time'] - cooling_list[0]['real_time']).total_seconds()
                    cooling_row["CcACCEnd"] = (
                                cooling_list[-1]['real_time'] - cooling_list[1]['real_time']).total_seconds()
                    cooling_row["CcTotal"] = (
                                cooling_list[-1]['real_time'] - cooling_list[0]['real_time']).total_seconds()
            else:
                cooling_row["CcDQEnd"] = 0
                cooling_row["CcACCEnd"] = 0
                cooling_row["CcTotal"] = 0
            # 有问题
            rolling_total = 0.0
            for dic_i in rolling_dic:
                rolling_total += rolling_dic[dic_i]
            # rolling_total = fm_list[-1]['real_time'] - rm_list[0]['real_time']
            rolling_dic['rolling_total'] = rolling_total
            rolling_rows.append(rolling_dic)
            cooling_rows.append(cooling_row)
            # 冷却
            # if
        heat_df = pd.DataFrame(data=heat_rows, columns=heat_col_names).dropna(axis=0, how='all').reset_index(drop=True)
        rolling_df = pd.DataFrame(data=rolling_rows, columns=rolling_cols).dropna(axis=0, how='all').reset_index(
            drop=True)
        cooling_df = pd.DataFrame(data=cooling_rows, columns=cooling_cols).dropna(axis=0, how='all').reset_index(
            drop=True)

        df_col = pd.concat([heat_df, rolling_df, cooling_df], axis=1)
        return df_col
