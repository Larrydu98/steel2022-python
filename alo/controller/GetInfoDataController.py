from ..models.GetInfoDataDB import GetInfoDataDB
import datetime as dt
from ..utils import readConfig, allGetSQLData
import numpy as np
import pandas as pd
import portion as P


class GetInfoDataController:
    def __init__(self, args, start_time, end_time, merge_limit, merge_conflict):
        res = GetInfoDataDB(start_time=start_time, end_time=end_time)
        self.merge_limit, self.merge_conflict = int(merge_limit), int(merge_conflict)
        self.post_table = args
        his_rows, his_col_names = res.getCurrentData()
        self.his_dataframe = pd.DataFrame(data=his_rows, columns=his_col_names).dropna(axis=0, how='all').reset_index(drop=True)

    def getInfoData(self):
        res = {}
        production_rhythm_list = []
        merge_list = self.getMergeList()
        for mer_index, mer_val in enumerate(merge_list):
            res['series' + str(mer_index + 1)] = {}
            merge_df = self.his_dataframe.loc[mer_val]
            process_df = processDataframe(merge_df)
            production_rhythm = (merge_df.iloc[-1]['toc'] - merge_df.iloc[0]['toc']).total_seconds() / len(merge_df)
            production_rhythm_list.append(production_rhythm)
            # max_list.append(process_df.max())
            # process_list.append(process_df)
            process_mean = process_df.mean()
            res['series' + str(mer_index + 1)]['production_rhythm'] = production_rhythm
            res['series' + str(mer_index + 1)]['heating_mean'] = [process_mean['heat1'], process_mean['heat2'],
                                                                  process_mean['heat3'], process_mean['heat4'],
                                                                  process_mean['heat5']]
            res['series' + str(mer_index + 1)]['rolling_mean'] = [process_mean['RmF3Pass'], process_mean['RmL3Pass'],
                                                                  process_mean['RmEnd'],process_mean['FmF3Pass'],
                                                                  process_mean['FmL3Pass'], process_mean['FmEnd']]
            res['series' + str(mer_index + 1)]['cooling_mean'] = [process_mean['CcDQEnd'], process_mean['CcACCEnd']]
            res['series' + str(mer_index + 1)]['total_mean'] = [process_mean['heat_total'],
                                                                process_mean['rolling_total'], process_mean['CcTotal']]
            res['series' + str(mer_index + 1)]['total_var'] = [process_df['heat_total'].var(),
                                                               process_df['rolling_total'].var(),
                                                               process_df['CcTotal'].var()]
        #     if mer_index == 0:
        #         max_df = process_df.max()
        #         continue
        #     else:
        #         max_df = pd.concat([max_df, process_df.max()], axis=1)
        # all_max = max_df.max(axis=1)
        # 添加元素
        # res['max_detail'] = {}
        # res['max_detail']['heating_max'] = [all_max['heat1'], all_max['heat2'], all_max['heat3'], all_max['heat4'],
        #                                     all_max['heat5']]
        # res['max_detail']['rolling_max'] = [all_max['RmF3Pass'], all_max['RmL3Pass'], all_max['RmEnd'],all_max['FmStart'],
        #                                     all_max['FmF3Pass'], all_max['FmL3Pass'], all_max['FmEnd']]
        # res['max_detail']['cooling__max'] = [all_max['CcDQEnd'], all_max['CcACCEnd']]
        # res['max_detail']['total_max'] = [max(production_rhythm_list), all_max['heat_total'], all_max['rolling_total'],
        #                                   all_max['CcTotal']]
        return res

    def getMergeList(self):
        his_dataframe = self.his_dataframe
        for index in self.post_table:
            if index == 'cooling':
                his_dataframe = his_dataframe[his_dataframe['status_cooling'] == self.post_table[index]]
                continue
            if index == 'platetype':
                his_dataframe = his_dataframe[his_dataframe[index] == self.post_table[index]]
                continue
            his_dataframe = his_dataframe[(his_dataframe[index] >= self.post_table[index][0]) & (
                        his_dataframe[index] < self.post_table[index][1])]
        his_index = his_dataframe.index.values.tolist()
        merge_list = self.judgeMerge(his_index)
        res = []
        for i in merge_list:
            if len(i) >= self.merge_limit:
                res.append(i)
        return res

    def judgeMerge(self, data):
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


# 获取种类中的加热轧制冷却的时间占比信息
def processDataframe(data):
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
        if len(rm_list) != 0:
            if len(rm_list) <= 3:
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
        else:
            continue
        # 精轧
        if len(fm_list) != 0:
            if len(fm_list) <= 3:
                FmF3Pass = fm_list[-1]['real_time'] - rm_list[0]['real_time']

                rolling_dic["FmF3Pass"] = FmF3Pass.total_seconds()
                rolling_dic["FmL3Pass"] = 0
                rolling_dic["FmEnd"] = 0
            elif len(fm_list) > 3 and len(fm_list) <= 6:

                FmF3Pass = fm_list[2]['real_time'] - fm_list[0]['real_time']
                FmL3Pass = fm_list[-1]['real_time'] - fm_list[2]['real_time']

                rolling_dic["FmF3Pass"] = FmF3Pass.total_seconds()
                rolling_dic["FmL3Pass"] = FmL3Pass.total_seconds()
                rolling_dic["FmEnd"] = 0
            elif len(fm_list) > 6:
                FmF3Pass = fm_list[2]['real_time'] - fm_list[0]['real_time']
                FmL3Pass = fm_list[-3]['real_time'] - fm_list[2]['real_time']
                FmEnd = fm_list[-1]['real_time'] - fm_list[-3]['real_time']

                rolling_dic["FmF3Pass"] = FmF3Pass.total_seconds()
                rolling_dic["FmL3Pass"] = FmL3Pass.total_seconds()
                rolling_dic["FmEnd"] = FmEnd.total_seconds()
        # 冷却
        if len(cooling_list) != 0:
            if len(cooling_list) == 2:
                cooling_row["CcDQEnd"] = (cooling_list[-1]['real_time'] - cooling_list[0]['real_time']).total_seconds()
                cooling_row["CcACCEnd"] = 0
                cooling_row["CcTotal"] = (cooling_list[-1]['real_time'] - cooling_list[0]['real_time']).total_seconds()
            else:
                cooling_row["CcDQEnd"] = (cooling_list[1]['real_time'] - cooling_list[0]['real_time']).total_seconds()
                cooling_row["CcACCEnd"] = (cooling_list[-1]['real_time'] - cooling_list[1]['real_time']).total_seconds()
                cooling_row["CcTotal"] = (cooling_list[-1]['real_time'] - cooling_list[0]['real_time']).total_seconds()
        else:
            cooling_row["CcDQEnd"] = 0
            cooling_row["CcACCEnd"] = 0
            cooling_row["CcTotal"] = 0
        rolling_total = fm_list[-1]['real_time'] - rm_list[0]['real_time']
        rolling_dic['rolling_total'] = rolling_total.total_seconds()
        rolling_rows.append(rolling_dic)
        cooling_rows.append(cooling_row)
        # 冷却
        # if
    heat_df = pd.DataFrame(data=heat_rows, columns=heat_col_names).dropna(axis=0, how='all').reset_index(drop=True)
    rolling_df = pd.DataFrame(data=rolling_rows, columns=rolling_cols).dropna(axis=0, how='all').reset_index(drop=True)
    cooling_df = pd.DataFrame(data=cooling_rows, columns=cooling_cols).dropna(axis=0, how='all').reset_index(drop=True)

    df_col = pd.concat([heat_df, rolling_df, cooling_df], axis=1)
    return df_col
