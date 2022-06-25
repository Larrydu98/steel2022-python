from ..models.ganttChartDataDB import getGanttChartDataDB
import pandas as pd
from ..utils import NewCluster
import datetime as dt
import numpy as np
import json
import math

class getGanttChartData():
    def __init__(self, start_time, end_time, minute_diff, merge_limit, merge_conflict, args, cooling):
        rows, col_names = getGanttChartDataDB.ganttChartDataDB(start_time, end_time)
        self.gantt_data = pd.DataFrame(data=rows, columns=col_names).dropna(axis=0, how='all').reset_index(drop=True)
        if True:
            cluster = NewCluster(self.gantt_data, minute_diff, merge_limit, merge_conflict, args, cooling)
            self.batch_plate, self.category_plate = cluster.getClusterData()

    def getGanttChartData(self):
        # conn = psycopg2.connect(database='BSData20190712', user='postgres', password='woshimima',
        #                         host='219.216.80.146', port='5432')
        # cursor = conn.cursor()
        # cursor = conn.cursor()

        # 批次划分的时间差
        res = []
        for bat_i, bat_val in enumerate(self.category_plate):

            # if bat_i == 42:
                # print(bat_i)
            for cat_i, cat_val in enumerate(bat_val):
                toc = str(self.category_plate[bat_i][cat_i]['data'].iloc[0]['toc'])
                bid = toc[2:4] + hex(int(toc[5:7])).upper()[2:] + toc[8:10] + toc[11:13]
                # smode = 'A'
                # smode = 'B'
                smode = 'C'


                df = self.category_plate[bat_i][cat_i]['data']
                cflag = 0 if self.category_plate[bat_i][cat_i]['merge_flag'] else 1
                # debugger
                # print(bat_i,cat_i)
                # if cat_i == 0:
                #     print('test')
                if cat_val['merge_flag']:
                    # print(cat_i)
                    # print('----------')
                    # cindex
                    cindex = statisticsAll.getCindex(df)
                    # pindex
                    # 质量异常状态
                    quaStatistics = statisticsAll.countAllflag(df)
                    # 规格指标统计计量
                    speStatistics = statisticsAll.getSpeStatistics(df)
                    maryDetail = statisticsAll.processDataframe(df)
                    pindex = {
                        # 质量异常状态
                        "quaStatistics": quaStatistics,
                        # 马雷转存数据
                        "maryDetail": maryDetail,
                        # 规格指标统计量
                        "speStatistics": speStatistics
                    }
                    upids = {}
                    timeToc = {}
                    for dic_index, dic_val in enumerate(cat_val['dataCluster']):
                        suoyin = str(int(dic_index + 1))

                        upids['upid' + suoyin] = dic_val.upid.tolist()
                        timeToc['time' + suoyin] = [dic_val.iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
                                                                dic_val.iloc[len(dic_val) - 1].toc.strftime(
                                                                    "%Y-%m-%d %H:%M:%S")]
                    # res.append([bid, str(cat_i + 1).zfill(2), smode, json.dumps(upids), json.dumps(timeToc), cflag, json.dumps(cindex),
                    #             json.dumps(pindex)])

                    res.append([bid, str(cat_i + 1).zfill(2), smode, upids, timeToc, cflag,
                                cindex,
                                pindex])

                else:
                    # res.append([bid, str(cat_i + 1).zfill(2), smode, json.dumps({"upid1": df.upid.tolist()}),
                    #             json.dumps({"time1": [df.iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"), df.iloc[len(df) - 1].toc.strftime("%Y-%m-%d %H:%M:%S")]}), cflag, json.dumps({}),
                    #             json.dumps({})])

                    res.append([bid, str(cat_i + 1).zfill(2), smode, {"upid1": df.upid.tolist()}, {
                        "time1": [df.iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
                                  df.iloc[len(df) - 1].toc.strftime("%Y-%m-%d %H:%M:%S")]}, cflag, {}, {}])

        column_names = ['bid', 'cid', 'smode', 'upids', 'time', 'cflag', 'cindex', 'pindex']
        df = pd.DataFrame(data=res, columns=column_names).dropna(axis=0, how='all').reset_index(drop=True)
        # output(df)
        # df.to_csv('E:\data\data1357.csv', sep=',', header=True, index=False)
        df.to_csv('E:\data\data1357.csv', sep='\t', header=False, index=False)

        # df.to_json('E:\data\data1357.json',)

        return 'res'
            # good_flag, bad_flag, no_flag = countGoodBadNoflag(self.batch_plate[bat_i], 'dataframe')
            # res.append({
            #     'batch': bat_i + 1,
            #     'good_flag': good_flag,
            #     'bad_flag': bad_flag,
            #     'no_flag': no_flag,
            #     'total': len(self.batch_plate[bat_i]),
            #     'startTime': self.batch_plate[bat_i].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
            #     'endTime': self.batch_plate[bat_i].iloc[len(self.batch_plate[bat_i]) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
            #     "tgtwidth_avg": round(self.batch_plate[bat_i].tgtwidth.mean(), 3),
            #     "tgtlength_avg": round(self.batch_plate[bat_i].tgtlength.mean(), 3),
            #     "tgtthickness_avg": round(self.batch_plate[bat_i].tgtthickness.mean(), 5),
            #     'category': []
            # })
            # for cat_i, cat_val in enumerate(bat_val):
            #     good_flag, bad_flag, no_flag = countGoodBadNoflag(cat_val['data'], 'dataframe')
            #     if cat_val['merge_flag']:
            #         res[bat_i]['category'].append({
            #             'merge_flag': cat_val['merge_flag'],
            #             'category_index': cat_i + 1,
            #             'platetype': cat_val['data'].iloc[0, :]['platetype'],
            #             'good_flag': good_flag,
            #             'bad_flag': bad_flag,
            #             'no_flag': no_flag,
            #             'total': len(cat_val['data']),
            #             'startTime': cat_val['data'].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
            #             'endTime': cat_val['data'].iloc[len(cat_val['data']) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
            #             "tgtwidth_avg": round(cat_val['data'].tgtwidth.mean(), 3),
            #             "tgtlength_avg": round(cat_val['data'].tgtlength.mean(), 3),
            #             "tgtthickness_avg": round(cat_val['data'].tgtthickness.mean(), 5),
            #             "tgtthickness_avg": round(cat_val['data'].tgtthickness.mean(), 5),
            #             'detail': []
            #         })
            #     else:
            #         res[bat_i]['category'].append({
            #             'merge_flag': cat_val['merge_flag'],
            #             'category_index': cat_val['category_index'],
            #             'platetype': cat_val['data'].iloc[0, :]['platetype'],
            #             'good_flag': good_flag,
            #             'bad_flag': bad_flag,
            #             'no_flag': no_flag,
            #             'total': len(cat_val['data']),
            #             'startTime': cat_val['data'].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
            #             'endTime': cat_val['data'].iloc[len(cat_val['data']) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
            #             "tgtwidth_avg": round(cat_val['data'].tgtwidth.mean(), 3),
            #             "tgtlength_avg": round(cat_val['data'].tgtlength.mean(), 3),
            #             "tgtthickness_avg": round(cat_val['data'].tgtthickness.mean(), 5),
            #             "tgtthickness_avg": round(cat_val['data'].tgtthickness.mean(), 5),
            #             'detail': []
            #         })
            #     for plate_i, plate_val in cat_val['data'].iterrows():
            #         res[bat_i]['category'][cat_i]['detail'].append({
            #             'upid': plate_val['upid'],
            #             'toc': plate_val['toc'].strftime("%Y-%m-%d %H:%M:%S"),
            #             'platetype': plate_val['platetype'],
            #             'tgtwidth': plate_val['tgtwidth'],
            #             'tgtlength': plate_val['tgtlength'],
            #             'tgtthickness': plate_val['tgtthickness'],
            #             'flag_lable': countGoodBadNoflag(plate_val,'single'),
            #             'status_cooling': plate_val['status_cooling']
            #         })
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

import psycopg2
def output(data):
    from io import StringIO
    # 连接数据库
    # data =
    conn = psycopg2.connect(database='BSData20190712', user='postgres', password='woshimima',
                                                    host='219.216.80.146', port='5432')

    # dataframe类型转换为IO缓冲区中的str类型
    output = StringIO()
    data.to_csv(output, sep='\t', index=False, header=False)
    output1 = output.getvalue()

    cur = conn.cursor()
    cur.copy_from(StringIO(output1), '"dbcTest".deba_dump_resave')

    conn.commit()
    cur.close()
    conn.close()
    print('finish!!!!')
    return True


class statisticsAll():
    # 统计cIndex的数据
    @staticmethod
    def getCindex(df):
        # 这里的是参数
        cindex = {}
        cindex['platetype'] = df.iloc[0].platetype
        # modeA
        # cindex['tgtthickness'] = [round(df.tgtthickness.min(), 5), round(df.tgtthickness.min() + 0.004, 5)]
        # cindex['tgtwidth'] = [round(df.tgtwidth.min(), 3), round(df.tgtwidth.min() + 2, 3)]
        # cindex['tgtlength'] = [round(df.tgtlength.min(), 3), round(df.tgtlength.min() + 60, 3)]

        # modeB
        # cindex['tgtthickness'] = [round(df.tgtthickness.min(), 5), round(df.tgtthickness.min() + 0.01, 5)]
        # cindex['tgtwidth'] = [round(df.tgtwidth.min(), 3), round(df.tgtwidth.min() + 4, 3)]
        # cindex['tgtlength'] = [round(df.tgtlength.min(), 3), round(df.tgtlength.min() + 100, 3)]

        # modeC
        cindex['tgtthickness'] = [round(df.tgtthickness.min(), 5), round(df.tgtthickness.min() + 0.002, 5)]
        cindex['tgtwidth'] = [round(df.tgtwidth.min(), 3), round(df.tgtwidth.min() + 1, 3)]
        cindex['tgtlength'] = [round(df.tgtlength.min(), 3), round(df.tgtlength.min() + 30, 3)]
        # cindex['tgtdischargetemp'] = [round(df.tgtdischargetemp.min(), 3), round(df.tgtdischargetemp.min() + 10, 3)]
        cindex['cooling'] = int(df.iloc[0].status_cooling)
        return cindex

    # 统计质量异常状态
    @staticmethod
    def countAllflag(data):
        good_flag = len(data[(data['status_fqc'] == 0) & (data['flag_lable'] == '[1, 1, 1, 1, 1]')])
        bad_flag = len(data[(data['status_fqc'] == 0) & (data['flag_lable'] != '[1, 1, 1, 1, 1]')])
        no_flag = len(data[data['status_fqc'] == 1])
        # 生产节奏
        # proRhythm = (data.iloc[-1]['discharge_time'] - data.iloc[0]['discharge_time']).total_seconds() / len(data)
        proRhythm = round( len(data) * 3600 / (data.iloc[-1]['discharge_time'] - data.iloc[0]['discharge_time']).total_seconds(), 2)

        newData = data.loc[data['status_fqc'] == 0]
        fiveFlag = np.zeros(5)
        abnormalName = ['bend','abnormalThickness', 'horizonWave', 'leftWave',  'rightWave', ]
        for index, val in newData.iterrows():
            flagArr = eval(val['flag_lable'])
            for i in range(5):
                if flagArr[i] == 0:
                    fiveFlag[i] += 1
        abnormalDetail = {}
        for flagI, flagVal in enumerate(fiveFlag):
            abnormalDetail[abnormalName[flagI]] = int(flagVal)
        res = {
            "goodNum": int(good_flag),
            "badNum": int(bad_flag),
            'noNum': int(no_flag),
            'abnormalDetail': abnormalDetail,
            # "bend": int(fiveFlag[0]),
            # "abnormalThickness": int(fiveFlag[1]),
            # "horizonWave": int(fiveFlag[2]),
            # "leftWave": int(fiveFlag[3]),
            # "rightWave": int(fiveFlag[4]),
            "proRhythm": proRhythm
        }
        return res

    # 统计马雷图中的circle中的数据
    @staticmethod
    def processDataframe(data):
        heat_col_names = ["heat1", "heat2", "heat3", "heat4", "heat5", "heat_total"]
        heat_rows = []
        rolling_cols = ['RmF3Pass', 'RmL3Pass', 'RmEnd', 'FmStart', 'FmF3Pass', 'FmL3Pass', 'FmEnd', 'rolling_total']
        rolling_rows = []
        cooling_cols = ['CcDQEnd', 'CcACCEnd', 'CcTotal']
        cooling_rows = []
        for plate_index, plate_val in data.iterrows():
            # print('plate_index',plate_index)
            if not plate_val['stops']:
                continue
            heat_list, rm_list, fm_list, cooling_list = [], [], [], []
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
                elif 'Fu' in stop_val['station']['name']:
                    heat_list.append({'index': stop_index, 'name': stop_val['station']['name'],
                                         'real_time': dt.datetime.strptime(stop_val['realTime'], '%Y-%m-%d %H:%M:%S'),
                                         'time': stop_val['time']})
            # if len(rm_list):
            #     continue
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
            rolling_dic = {}
            cooling_row = {}


            # 粗轧
            if len(rm_list) == 0:
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
            if len(fm_list) == 0 :
                rolling_dic["FmStart"] = 0
                rolling_dic["FmF3Pass"] = 0
                rolling_dic["FmL3Pass"] = 0
                rolling_dic["FmEnd"] = 0
            elif len(fm_list) <= 3 and len(fm_list) > 0:
                if len(rm_list) == 0:
                    FmStart = fm_list[0]['real_time'] - heat_list[-1]['real_time']
                else:
                    FmStart = fm_list[0]['real_time'] - rm_list[-1]['real_time']
                FmF3Pass = fm_list[-1]['real_time'] - fm_list[0]['real_time']

                rolling_dic["FmStart"] = FmStart.total_seconds()
                rolling_dic["FmF3Pass"] = FmF3Pass.total_seconds()
                rolling_dic["FmL3Pass"] = 0
                rolling_dic["FmEnd"] = 0
            elif len(fm_list) > 3 and len(fm_list) <= 6:
                if len(rm_list) == 0:
                    FmStart = fm_list[0]['real_time'] - heat_list[-1]['real_time']
                else:
                    FmStart = fm_list[0]['real_time'] - rm_list[-1]['real_time']
                FmF3Pass = fm_list[2]['real_time'] - fm_list[0]['real_time']
                FmL3Pass = fm_list[-1]['real_time'] - fm_list[2]['real_time']

                rolling_dic["FmStart"] = FmStart.total_seconds()
                rolling_dic["FmF3Pass"] = FmF3Pass.total_seconds()
                rolling_dic["FmL3Pass"] = FmL3Pass.total_seconds()
                rolling_dic["FmEnd"] = 0
            elif len(fm_list) > 6:
                if len(rm_list) == 0:
                    FmStart = fm_list[0]['real_time'] - heat_list[-1]['real_time']
                else:
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

        df_col = pd.concat([heat_df, rolling_df, cooling_df], axis=1).round(2)
        df_mean = df_col.mean()
        res = {}

        def test(x):
            if math.isnan(x):
                return -1
            else:
                return np.float(x)
        res['heating_mean'] = [df_mean['heat1'], df_mean['heat2'], df_mean['heat3'], df_mean['heat4'], df_mean['heat5']]
        res['rolling_mean'] = [df_mean['RmF3Pass'], df_mean['RmL3Pass'], df_mean['RmEnd'], df_mean['FmStart'],
                               df_mean['FmF3Pass'], df_mean['FmL3Pass'], df_mean['FmEnd']]
        res['cooling_mean'] = [df_mean['CcDQEnd'], df_mean['CcACCEnd']]
        res['total_mean'] = [df_mean['heat_total'], df_mean['rolling_total'], df_mean['CcTotal']]
        res['total_var'] = [df_col['heat_total'].var(), df_col['rolling_total'].var(), df_col['CcTotal'].var()]
        res['heating_mean'] = list(map(test, res['heating_mean']))
        res['rolling_mean'] = list(map(test, res['rolling_mean']))
        res['cooling_mean'] = list(map(test, res['cooling_mean']))
        res['total_mean'] = list(map(test, res['total_mean']))
        res['total_var'] = list(map(test, res['total_var']))

        return res

    # 获取规格指标统计量 长宽高的平均值和标准差
    @staticmethod
    def getSpeStatistics(data):
        speStatistics = {
            'tgtwidth_avg': round(data.tgtwidth.mean(), 3),
            'tgtthickness_avg': round(data.tgtthickness.mean(), 5),
            'tgtlength_avg': round(data.tgtlength.mean(), 3),
            # 'tgttmplatetemp_avg': data.tgttmplatetemp.mean(),
            # 'tgtdischargetemp_avg': data.tgtdischargetemp.mean(),
            'tgtwidth_std': round(data.tgtwidth.std(), 3),
            'tgtthickness_std': round(data.tgtthickness.std(), 5),
            'tgtlength_std': round(data.tgtlength.std(), 3),
            # 'tgttmplatetemp_std': data.tgttmplatetemp.std(),
            # 'tgtdischargetemp_std': data.tgtdischargetemp.std()
        }
        return speStatistics
