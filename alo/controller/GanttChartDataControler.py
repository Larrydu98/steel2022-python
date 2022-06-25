from ..models.ganttChartDataDB import getGanttChartDataDB
import pandas as pd
from ..utils import countGoodBadNoflag, Cluster
import datetime as dt




class getGanttChartData():
    def __init__(self, start_time, end_time,minute_diff, merge_limit, merge_conflict, args, cooling):
        rows, col_names = getGanttChartDataDB.ganttChartDataDB(start_time, end_time)
        self.gantt_data = pd.DataFrame(data=rows, columns=col_names).dropna(axis=0, how='all').reset_index(drop=True)
        if True:
            cluster = Cluster(self.gantt_data, minute_diff, merge_limit, merge_conflict, args, cooling)
            self.batch_plate, self.category_plate = cluster.getClusterData()

    def getGanttChartData(self):
        # 批次划分的时间差
        res = []
        result = []
        for bat_i, bat_val in enumerate(self.category_plate):
            good_flag, bad_flag, no_flag = countGoodBadNoflag(self.batch_plate[bat_i], 'dataframe')
            res.append({
                'batch': bat_i + 1,
                'good_flag': good_flag,
                'bad_flag': bad_flag,
                'no_flag': no_flag,
                'total': len(self.batch_plate[bat_i]),
                'startTime': self.batch_plate[bat_i].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
                'endTime': self.batch_plate[bat_i].iloc[len(self.batch_plate[bat_i]) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
                "tgtwidth_avg": round(self.batch_plate[bat_i].tgtwidth.mean(), 3),
                "tgtlength_avg": round(self.batch_plate[bat_i].tgtlength.mean(), 3),
                "tgtthickness_avg": round(self.batch_plate[bat_i].tgtthickness.mean(), 5),
                'category': []
            })
            for cat_i, cat_val in enumerate(bat_val):
                good_flag, bad_flag, no_flag = countGoodBadNoflag(cat_val['data'], 'dataframe')
                if cat_val['merge_flag']:
                    res[bat_i]['category'].append({
                        'merge_flag': cat_val['merge_flag'],
                        'category_index': cat_i + 1,
                        'platetype': cat_val['data'].iloc[0, :]['platetype'],
                        'good_flag': good_flag,
                        'bad_flag': bad_flag,
                        'no_flag': no_flag,
                        'total': len(cat_val['data']),
                        'startTime': cat_val['data'].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
                        'endTime': cat_val['data'].iloc[len(cat_val['data']) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
                        "tgtwidth_avg": round(cat_val['data'].tgtwidth.mean(), 3),
                        "tgtlength_avg": round(cat_val['data'].tgtlength.mean(), 3),
                        "tgtthickness_avg": round(cat_val['data'].tgtthickness.mean(), 5),
                        "tgtthickness_avg": round(cat_val['data'].tgtthickness.mean(), 5),
                        'detail': []
                    })
                else:
                    res[bat_i]['category'].append({
                        'merge_flag': cat_val['merge_flag'],
                        'category_index': cat_val['category_index'],
                        'platetype': cat_val['data'].iloc[0, :]['platetype'],
                        'good_flag': good_flag,
                        'bad_flag': bad_flag,
                        'no_flag': no_flag,
                        'total': len(cat_val['data']),
                        'startTime': cat_val['data'].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
                        'endTime': cat_val['data'].iloc[len(cat_val['data']) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
                        "tgtwidth_avg": round(cat_val['data'].tgtwidth.mean(), 3),
                        "tgtlength_avg": round(cat_val['data'].tgtlength.mean(), 3),
                        "tgtthickness_avg": round(cat_val['data'].tgtthickness.mean(), 5),
                        "tgtthickness_avg": round(cat_val['data'].tgtthickness.mean(), 5),
                        'detail': []
                    })
                for plate_i, plate_val in cat_val['data'].iterrows():
                    res[bat_i]['category'][cat_i]['detail'].append({
                        'upid': plate_val['upid'],
                        'toc': plate_val['toc'].strftime("%Y-%m-%d %H:%M:%S"),
                        'platetype': plate_val['platetype'],
                        'tgtwidth': plate_val['tgtwidth'],
                        'tgtlength': plate_val['tgtlength'],
                        'tgtthickness': plate_val['tgtthickness'],
                        'flag_lable': countGoodBadNoflag(plate_val,'single'),
                        'status_cooling': plate_val['status_cooling']
                    })
        # for index, value in enumerate(self.category_plate ):
        #
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
        return res

class statisticsAll():
    def __init__(self, df):
        self.data = df
    @staticmethod
    def countAllflag(self):
        good_flag = len(self.data [(self.data ['status_fqc'] == 0) & (self.data ['flag_lable'] == '[1, 1, 1, 1, 1]')])
        bad_flag = len(self.data [(self.data ['status_fqc'] == 0) & (self.data ['flag_lable'] != '[1, 1, 1, 1, 1]')])
        no_flag = len(self.data [self.data ['status_fqc'] == 1])
        return good_flag, bad_flag, no_flag
    @staticmethod
    def processDataframe(self):
        heat_col_names = ["heat1", "heat2", "heat3", "heat4", "heat5", "heat_total"]
        heat_rows = []
        rolling_cols = ['RmF3Pass', 'RmL3Pass', 'RmEnd', 'FmStart', 'FmF3Pass', 'FmL3Pass', 'FmEnd', 'rolling_total']
        rolling_rows = []
        cooling_cols = ['CcDQEnd', 'CcACCEnd', 'CcTotal']
        cooling_rows = []
        for plate_index, plate_val in self.data.iterrows():
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
        production_rhythm = (self.data.iloc[-1]['toc'] - self.data.iloc[0]['toc']).total_seconds() / len(self.data)
        res = {}
        res['heating_mean'] = [df_col['heat1'], df_col['heat2'],
                                                                  df_col['heat3'], df_col['heat4'],
                                                                  df_col['heat5']],;
        res['rolling_mean'] = [df_col['RmF3Pass'], df_col['RmL3Pass'], df_col['RmEnd'], df_col['FmStart'], df_col['FmF3Pass'],df_col['FmL3Pass'], df_col['FmEnd']],
        res['cooling_mean'] = [df_col['CcDQEnd'], df_col['CcACCEnd']],
        res['total_mean'] = [df_col['heat_total'], df_col['rolling_total'], df_col['CcTotal']],
        res['total_var'] = [df_col['heat_total'].var(), df_col['rolling_total'].var(), df_col['CcTotal'].var()]

        return res, production_rhythm

