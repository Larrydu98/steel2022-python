import json
import psycopg2
import numpy as np
import pandas as pd
import datetime as dt

from ..models.getMareyDataFromDB import *


class ComputeMareyData:
    def __init__(self, type, upid, start_time, end_time, steelspec, tgtplatethickness):

        rows, col_names = GetMareyData.getMareyData(type, upid, start_time, end_time, steelspec, tgtplatethickness)

        self.marey_data = pd.DataFrame(data=rows, columns=col_names).dropna(axis=0, how='all').reset_index(drop=True)
        # self.marey_data.cooling_start_temp = self.marey_data.cooling_start_temp.replace('None',0)
        # self.marey_data.cooling_stop_temp = self.marey_data.cooling_stop_temp.replace('None',0)
        # self.marey_data.cooling_rate1 = self.marey_data.cooling_rate1.replace('None',0)
        if(type=="stations"):
            index = pd.MultiIndex.from_arrays([self.marey_data.upid.values.tolist()], names=['upid'])
        else:
            index = pd.MultiIndex.from_arrays([self.marey_data.upid.values.tolist(), self.marey_data.pass_no.values.tolist()], names=['upid', 'pass_no'])
        self.marey_data.index = index

        # print(len(self.marey_data))

    def printData(self):
        print(self.marey_data)

    def getMareyStations(self):
        if len(self.marey_data) == 0:
            return 204, []

        mareystations = []
        mareystations.append({'key': '0101', 'name': 'FuCharging', 'distance': 0.0, 'zone': '1'})
        mareystations.append({'key': '0102', 'name': 'FuPre', 'distance': 40.0, 'zone': '1'})
        mareystations.append({'key': '0103', 'name': 'FuHeating1', 'distance': 80.0, 'zone': '1'})
        mareystations.append({'key': '0104', 'name': 'FuHeating2', 'distance': 120.0, 'zone': '1'})
        mareystations.append({'key': '0105', 'name': 'FuSoak', 'distance': 160.0, 'zone': '1'})
        mareystations.append({'key': '0106', 'name': 'FuDischarging', 'distance': 200.0, 'zone': '1'})

        rm_pass_max = self.marey_data.totalpassesrm.max()
        fm_pass_max = self.marey_data.totalpassesfm.max()
        m_total_pass_max = rm_pass_max + fm_pass_max
        acc_flag = int(self.marey_data.adcontrolcode.max())
        cc_name = ['CcStart', 'CcDQEnd', 'CcACCEnd']

        for i in range(int(rm_pass_max)):
            mareystations.append({'key': '020' + str(i + 1),
                                  'name': 'RMPass' + str(i + 1),
                                  'distance': 200.0 + (i + 1) * 40,
                                  'zone': '2'})

        for i in range(int(fm_pass_max)):
            mareystations.append({'key': '030' + str(i + 1),
                                  'name': 'FMPass' + str(i + 1),
                                  'distance': 200.0 + (rm_pass_max + i + 1) * 40,
                                  'zone': '3'})
        if acc_flag > 0:
            for i in range(len(cc_name)):
                mareystations.append({'key': '040' + str(i + 1),
                                      'name': cc_name[i],
                                      'distance': 200.0 + (rm_pass_max + fm_pass_max + i + 1) * 40,
                                      'zone': '4'})

        return 200, mareystations


    def getMareyTimes(self, compressed_factor):

        if len(self.marey_data) == 0:
            return 204, []

        mareydata = []
        upids = self.marey_data.upid.unique()
        # print(upids)

        for i in range(len(upids)):
            # 参数初始化
            stops = []
            ## 压缩系数
            compressed_factor = int(compressed_factor)
            plate_data = self.marey_data.loc[upids[i]].loc[1]
            ## 去除重复道次数据
            if plate_data.shape != (31,):
                plate_data = self.marey_data.loc[upids[i]].loc[1].iloc[0,]
            rm_pass = plate_data.totalpassesrm
            fm_pass = plate_data.totalpassesfm
            m_pass_series = self.marey_data[self.marey_data.upid == upids[i]].pass_no
            rm_pass_ndarray = np.unique(m_pass_series[m_pass_series <= rm_pass].values)
            fm_pass_ndarray = np.unique(m_pass_series[m_pass_series > rm_pass].values)
            m_total_pass = len(rm_pass_ndarray) + len(fm_pass_ndarray)
            rm_pass_max = self.marey_data.totalpassesrm.max()
            fm_pass_max = self.marey_data.totalpassesfm.max()
            m_total_pass_max = rm_pass_max + fm_pass_max
            cc_name = ['CcStart', 'CcDQEnd', 'CcACCEnd']

            # 加热工序站点时间计算
            ## realTime
            fu_discharge_time = plate_data.discharge_time
            fu_charging_time = plate_data.discharge_time - dt.timedelta(minutes=plate_data.in_fce_time)
            staying_time_pre = fu_charging_time + dt.timedelta(minutes=plate_data.staying_time_pre)
            staying_time_1 = staying_time_pre + dt.timedelta(minutes=plate_data.staying_time_1)
            staying_time_2 = staying_time_1 + dt.timedelta(minutes=plate_data.staying_time_2)
            staying_time_soak = staying_time_2 + dt.timedelta(minutes=plate_data.staying_time_soak)
            ## time
            c_fu_discharge_time = fu_discharge_time
            c_staying_time_pre = c_fu_discharge_time - (fu_discharge_time - staying_time_pre) / compressed_factor
            c_staying_time_1 = c_fu_discharge_time - (fu_discharge_time - staying_time_1) / compressed_factor
            c_staying_time_2 = c_fu_discharge_time - (fu_discharge_time - staying_time_2) / compressed_factor
            c_staying_time_soak = c_fu_discharge_time - (fu_discharge_time - staying_time_soak) / compressed_factor
            c_fu_charging_time = c_fu_discharge_time - (fu_discharge_time - fu_charging_time) / compressed_factor
            # 数据储存
            stops.append({'station': {'key': '0101', 'name': 'FuCharging', 'distance': 0.0, 'zone': '1'}, 'realTime': str(fu_charging_time),
                          'time': str(c_fu_charging_time)[0:19]})
            stops.append({'station': {'key': '0102', 'name': 'FuPre', 'distance': 40.0, 'zone': '1'}, 'realTime': str(staying_time_pre),
                          'time': str(c_staying_time_pre)[0:19]})
            stops.append({'station': {'key': '0103', 'name': 'FuHeating1', 'distance': 80.0, 'zone': '1'}, 'realTime': str(staying_time_1),
                          'time': str(c_staying_time_1)[0:19]})
            stops.append({'station': {'key': '0104', 'name': 'FuHeating2', 'distance': 120.0, 'zone': '1'}, 'realTime': str(staying_time_2),
                          'time': str(c_staying_time_2)[0:19]})
            stops.append({'station': {'key': '0105', 'name': 'FuSoak', 'distance': 160.0, 'zone': '1'}, 'realTime': str(staying_time_soak),
                          'time': str(c_staying_time_soak)[0:19]})
            stops.append({'station': {'key': '0106', 'name': 'FuDischarging', 'distance': 200.0, 'zone': '1'}, 'realTime': str(fu_discharge_time),
                          'time': str(c_fu_discharge_time)[0:19]})

            # 粗轧工序站点时间计算
            for j in range(len(rm_pass_ndarray)):
                rm_data = self.marey_data.loc[upids[i]].loc[rm_pass_ndarray[j]]
                ## 去除重复道次数据
                if rm_data.shape != (31,):
                    rm_data = self.marey_data.loc[upids[i]].loc[rm_pass_ndarray[j]].iloc[0,]
                rm_time = dt.datetime.strptime(rm_data.starttime[0:14], "%Y%m%d%H%M%S")
                ## 数据储存
                stops.append({'station': {'key': '020' + str(rm_pass_ndarray[j]),
                                          'name': 'RMPass' + str(rm_pass_ndarray[j]),
                                          'distance': 200.0 + (j + 1) * 40,
                                          'zone': '2'},
                              'realTime': str(rm_time),
                              'time': str(rm_time)})

            # 精轧工序站点时间计算
            for j in range(len(fm_pass_ndarray)):
                fm_data = self.marey_data.loc[upids[i]].loc[fm_pass_ndarray[j]]
                ## 去除重复道次数据
                if fm_data.shape != (31,):
                    fm_data = self.marey_data.loc[upids[i]].loc[fm_pass_ndarray[j]].iloc[0,]
                fm_time = dt.datetime.strptime(fm_data.starttime[0:14], "%Y%m%d%H%M%S")
                ## 数据储存
                stops.append({'station': {'key': '030' + str(fm_pass_ndarray[j] - rm_pass),
                                          'name': 'FMPass' + str(fm_pass_ndarray[j] - rm_pass),
                                          'distance': 200.0 + (rm_pass_max + j + 1) * 40,
                                          'zone': '3'},
                              'realTime': str(fm_time),
                              'time': str(fm_time)})

            # 冷却工序站点时间计算
            if int(plate_data.adcontrolcode) <= 0:
                mareydata.append({'upid': plate_data.upid,
                                  'slabid': plate_data.slabid,
                                  'toc': str(plate_data.toc),
                                  'steelspec': plate_data.steelspec,
                                  'productcategory': plate_data.productcategory,

                                  'tgtdischargetemp': plate_data.tgtdischargetemp,
                                  'slabthickness': plate_data.slabthickness,
                                  'tgtplatethickness': plate_data.tgtplatethickness,
                                  'tgtwidth': plate_data.tgtwidth,
                                  'tgtplatelength2': plate_data.tgtplatelength2,
                                  'tgttmplatetemp': plate_data.tgttmplatetemp,
                                  'adcontrolcode': plate_data.adcontrolcode,
                                  # 'cooling_start_temp': plate_data.cooling_start_temp,
                                  # 'cooling_stop_temp': plate_data.cooling_stop_temp,
                                  # 'cooling_rate1': plate_data.cooling_rate1,

                                  'stops': stops
                                  })

            else:
                ## cc_start、cc_end时间计算

                if pd.isnull(plate_data.avg_time_b):
                    plate_data.avg_time_b = 0
                if pd.isnull(plate_data.avg_time_w):
                    plate_data.avg_time_w = 0
                cooling_start_time = dt.datetime.strptime(plate_data.timerollingfinish[0:14], "%Y%m%d%H%M%S")
                cc_start = cooling_start_time + dt.timedelta(seconds=plate_data.avg_time_b)
                cc_end = cc_start + dt.timedelta(seconds=plate_data.avg_time_w)

                ## dq_end时间计算
                dq_ratio = plate_data.dq_count / (plate_data.dq_count + plate_data.acc_count)
                if pd.isnull(dq_ratio):
                    dq_ratio = 0
                dq_end = cc_start + (cc_end - cc_start) * dq_ratio
                # cc_start = str(cc_start)[0:19]
                # cc_end = str(cc_end)[0:19]

                ## 数据储存
                stops.append({'station': {'key': '0401',
                                          'name': cc_name[0],
                                          'distance': 200.0 + (m_total_pass_max + 1) * 40,
                                          'zone': '4'},
                              'realTime': str(cc_start)[0:19],
                              'time': str(cc_start)[0:19]})
                stops.append({'station': {'key': '0402',
                                          'name': cc_name[1],
                                          'distance': 200.0 + (m_total_pass_max + 2) * 40,
                                          'zone': '4'},
                              'realTime': str(dq_end)[0:19],
                              'time': str(dq_end)[0:19]})
                stops.append({'station': {'key': '0402',
                                          'name': cc_name[2],
                                          'distance': 200.0 + (m_total_pass_max + 3) * 40,
                                          'zone': '4'},
                              'realTime': str(cc_end)[0:19],
                              'time': str(cc_end)[0:19]})

                if pd.isnull(plate_data.cooling_start_temp) or pd.isnull(plate_data.cooling_stop_temp) or pd.isnull(plate_data.cooling_rate1):
                    plate_data.cooling_start_temp = 0
                    plate_data.cooling_stop_temp = 0
                    plate_data.cooling_rate1 = 0
                # 规格数据、全流程站点数据汇总
                mareydata.append({'upid': plate_data.upid,
                                  'slabid': plate_data.slabid,
                                  'toc': str(plate_data.toc),
                                  'steelspec': plate_data.steelspec,
                                  'productcategory': plate_data.productcategory,

                                  'tgtdischargetemp': plate_data.tgtdischargetemp,
                                  'slabthickness': plate_data.slabthickness,
                                  'tgtplatethickness': plate_data.tgtplatethickness,
                                  'tgtwidth': plate_data.tgtwidth,
                                  'tgtplatelength2': plate_data.tgtplatelength2,
                                  'tgttmplatetemp': plate_data.tgttmplatetemp,
                                  'adcontrolcode': plate_data.adcontrolcode,
                                  'cooling_start_temp': plate_data.cooling_start_temp,
                                  'cooling_stop_temp': plate_data.cooling_stop_temp,
                                  'cooling_rate1': plate_data.cooling_rate1,

                                  'stops': stops
                                  })

        if len(mareydata) < 5:
            return 202, mareydata
        return 200, mareydata
