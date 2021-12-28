import pandas as pd
import numpy as np
from ..api.singelSteel import filterSQL, getLabelData, new_filterSQL
from scipy.interpolate import interp1d
lefttable = ''' from  dcenter.l2_m_primary_data lmpd
            left join dcenter.l2_fu_acc_t lfat on lmpd.slabid = lfat.slab_no
            left join dcenter.l2_m_plate lmp   on lmpd.slabid = lmp.slabid
            left join dcenter.l2_cc_pdi lcp    on lmpd.slabid = lcp.slab_no
            left join app.deba_dump_data dd   on dd.upid = lmp.upid '''


class GetProcessVisualizationData:
    def __init__(self, parser, start_time, end_time, process, deviation, limitation):
        self.process = process
        self.deviation = 100 * float(deviation)
        self.limitation = limitation
        self.limit = 5

        SQL, status_cooling, fqcflag = new_filterSQL(parser)
        selection = []
        ismissing = []
        if (process == 'cool'):
            selection = ["dd.cooling", 'dd.fqc_label', 'dd.status_cooling', 'dd.upid']
            if fqcflag == 0:
                ismissing = "dd.status_cooling = 0 and dd.status_fqc = 0"
            elif fqcflag == 1:
                ismissing = "dd.status_cooling = 0"
        if (process == 'heat'):
            selection = ["dd.furnace", 'dd.fqc_label', 'dd.status_furnace', 'dd.upid']
            if fqcflag == 0:
                ismissing = "dd.status_furnace = 0 and dd.status_fqc = 0"
            elif fqcflag == 1:
                ismissing = "dd.status_furnace = 0"
        if (process == 'roll'):
            selection = ["dd.rolling", 'dd.fqc_label', 'dd.status_rolling', 'dd.upid']
            if fqcflag == 0:
                ismissing = "dd.status_rolling = 0 and dd.status_fqc = 0"
            elif fqcflag == 1:
                ismissing = "dd.status_rolling = 0"
        select = ','.join(selection)
        if (SQL != ''):
            SQL += ' and ' + ismissing
        if (SQL == ''):
            SQL = "where " + ismissing
        SQL = 'select ' + select + lefttable + SQL + ' ORDER BY dd.toc  DESC Limit ' + str(limitation)
        # print(SQL)
        data, col_names = getLabelData(SQL)
        # print(len(data))
        goodData = []

        if fqcflag == 0:
            for item in data:
                flags = item[1]['method1']['data']
                if np.array(flags).sum() == 5:
                    goodData.append(item)
        elif fqcflag == 1:
            goodData = data

        # print(len(goodData))
        self.data = pd.DataFrame(data=goodData, columns=col_names).dropna(axis=0, how='any').reset_index(drop=True)
        # print(data)

        # data = SQLplateselect(selection1, ismissing, tgtwidthSelect, tgtlengthSelect, tgtthicknessSelect, [], [], Platetypes, 'toc', '1000')
        sampleSQL = '''
                    select {select} 
                    {lefttable} 
                    where 
                        dd.toc >= '{start_time}'::timestamp and dd.toc <= '{end_time}'::timestamp;
                '''.format(select=select, lefttable=lefttable, start_time=start_time, end_time=end_time)
        sampledata, col_names = getLabelData(sampleSQL)
        self.sampledata = pd.DataFrame(data=sampledata, columns=col_names).dropna(axis=0, how='any').reset_index(drop=True)

    def getRoll(self):
        if len(self.data) == 0:
            return 204, {}

        names = ["bendingforce", "bendingforcebot", "bendingforcetop", "rollforce", "rollforceds", "rollforceos",
                "screwdown", "shiftpos", "speed", "torque", "torquebot", "torquetop"]
        name1 = ["contactlength", "entrytemperature", "exitflatness", "exitprofile", "exittemperature",
                 "exitthickness", "exitwidth", "forcecorrection"]

        serchkey = 'meas'
        for name in names:
            self.data[name] = self.data.apply(lambda x: np.array(x[0][serchkey][name]).mean(axis=1).tolist(), axis=1)
            self.sampledata[name] = self.sampledata.apply(lambda x: np.array(x[0][serchkey][name]).mean(axis=1).tolist(), axis=1)
        self.data["passcount"] = self.data.apply(lambda x: len(x["bendingforce"]), axis=1)
        self.sampledata["passcount"] = self.sampledata.apply(lambda x: len(x["bendingforce"]), axis=1)
        self.data.drop(columns="rolling", inplace=True)
        self.sampledata.drop(columns="rolling", inplace=True)

        all_pass = self.sampledata["passcount"].unique().tolist()
        _columns_result = {}
        for name in names:
            _pass_result = []
            for _pass in all_pass:
                train_pass = self.data[self.data["passcount"] == _pass][name]
                sample_pass = self.sampledata[self.sampledata["passcount"] == _pass][name]
                sample_upid = self.sampledata[self.sampledata["passcount"] == _pass]["upid"].values.tolist()
                if len(train_pass) == 0:
                    continue

                train_pass = np.array(train_pass.values.tolist())
                sample_pass = np.array(sample_pass.values.tolist())
                steel = {"min": [], "max": [], "emin": [], "emax": [], "mean": [], 'sample': [], "range": []}
                for p in range(_pass):  # p 为第几道次
                    _pass_i_train = train_pass[:, p]
                    _pass_i_sample = sample_pass[:, p]

                    # 数据过滤 / 处理
                    num = _pass_i_sample[_pass_i_sample > 0].shape[0]   # 当前道次sample大于0的钢板数量
                    if num == len(_pass_i_sample):
                        status = 1
                    elif num == 0:
                        status = -1
                    else:
                        status = 0

                    if name == "shiftpos":
                        if status > 0:
                            filter_rdata = _pass_i_train[_pass_i_train > 0]
                        if status < 0:
                            filter_rdata = _pass_i_train[_pass_i_train < 0]
                        steel["sample"].append(self.appendRollSampleData(np.absolute(_pass_i_sample), sample_upid))
                    else:
                        if status > 0:
                            filter_rdata = _pass_i_train[_pass_i_train > 0]
                        if status < 0:
                            filter_rdata = np.absolute(_pass_i_train)
                        steel["sample"].append(self.appendRollSampleData(_pass_i_sample, sample_upid))


                    if len(filter_rdata) != 0:
                        steel["min"].append(np.percentile(filter_rdata, self.deviation, axis=0))
                        steel["max"].append(np.percentile(filter_rdata, 100 - self.deviation, axis=0))
                        steel["emin"].append(np.percentile(filter_rdata, self.limit, axis=0))
                        steel["emax"].append(np.percentile(filter_rdata, 100 - self.deviation, axis=0))
                        steel["mean"].append(np.percentile(filter_rdata, 50, axis=0))
                _pass_result.append({
                    "passcount": p + 1,
                    "result": steel
                })
            _columns_result[name] = _pass_result
        return 200, _columns_result

    def getHeat(self, box_num):
        if len(self.data) == 0:
            return 204, {}

        names = ['seg_u', 'seg_d', 'plate', 'time']
        for name in names:
            self.data[name] = self.data.apply(lambda x: x[0][name], axis=1)
            self.sampledata[name] = self.sampledata.apply(lambda x: x[0][name], axis=1)
        self.data['position'] = self.data.apply(lambda x: x[0]['position'], axis=1)
        self.sampledata['position'] = self.sampledata.apply(lambda x: x[0]['position'], axis=1)
        self.data.drop(columns='furnace', inplace=True)
        self.sampledata.drop(columns='furnace', inplace=True)

        pos_range = self.heat_split_range(box_num)

        result = {}
        for name in names:
            data = [[] for _ in range(len(pos_range))]
            for i in range(len(self.data)):
                plate_data = self.data.iloc[i, :]
                for index, value in enumerate(plate_data['position']):
                    # 判断 value 在哪个区间
                    for j in range(len(pos_range)):
                        if j == 0 and value < pos_range[j][1]:
                            data[j].append(plate_data[name][j])
                            break
                        elif j == len(pos_range) - 1 and value > pos_range[j][1]:
                            data[j].append(plate_data[name][j])
                            break
                        elif value > pos_range[j][0] and value < pos_range[j][1]:
                            data[j].append(plate_data[name][j])
                            break

            sample_data = [[] for _ in range(len(pos_range))]
            for i in range(len(self.sampledata)):
                plate_data = self.sampledata.iloc[i, :]
                for index, value in enumerate(plate_data['position']):
                    # 判断 value 在哪个区间
                    for j in range(len(pos_range)):
                        if j == 0 and value < pos_range[j][1]:
                            sample_data[j].append({
                                'upid': plate_data.upid,
                                'value': plate_data[name][j],
                                'position': value
                            })
                            break
                        elif j == len(pos_range) - 1 and value > pos_range[j][1]:
                            sample_data[j].append({
                                'upid': plate_data.upid,
                                'value': plate_data[name][j],
                                'position': value
                            })
                            break
                        elif value > pos_range[j][0] and value < pos_range[j][1]:
                            sample_data[j].append({
                                'upid': plate_data.upid,
                                'value': plate_data[name][j],
                                'position': value
                            })
                            break

            result[name] = self.percentile(data, sample_data, self.deviation, self.limit)

        return 200, result

    def getCool(self, box_num, divide_percent):
        if len(self.data) == 0:
            return 204, {}
        return 200, {}

    def heat_split_range(self, box_num):
        min_list = []
        max_list = []
        for i in range(len(self.sampledata)):
            position = self.sampledata.iloc[i, :].position
            min_list.append(min(position))
            max_list.append(max(position))
        for i in range(len(self.data)):
            position = self.data.iloc[i, :].position
            min_list.append(min(position))
            max_list.append(max(position))
        min_pos = min(min_list)
        max_pos = max(max_list)

        pos_range = []
        point = [min_pos, 10, 22, 33, max_pos]
        for i, element in enumerate(point):
            if i == len(point) - 1:
                continue
            span = (point[i + 1] - point[i]) / box_num
            for j in range(box_num):
                pos_range.append([point[i] + span * j, point[i] + span * (j + 1)])

        return pos_range

    def appendRollSampleData(self, data, upid):
        res = []
        for i in range(len(data)):
            res.append({
                "upid": upid[i],
                "value": data[i]
            })
        return res

    def eyearray(self, tempnum):
        return np.linspace(0, 10 * np.pi, num=tempnum)

    def scipyutils(self, num, x_diff):
        x = self.eyearray(num)
        y = x_diff
        x_diff = self.eyearray(len(x_diff))
        f1 = interp1d(x_diff, y, kind='linear')  # 线性插值
        # f2=interp1d(x_diff,y,kind='cubic')#三次样条插值
        return f1(x)

    def percentile(self, datas, sample_datas, middeviation, exdeviation):
        steel = []
        for i, data in enumerate(datas):
            steel.append({
                "min": np.percentile(data, middeviation, axis=0),
                "max": np.percentile(data, 100 - middeviation, axis=0),
                "emin": np.percentile(data, exdeviation, axis=0),
                "emax": np.percentile(data, 100 - exdeviation, axis=0),
                "mean": np.percentile(data, 50, axis=0),
                "sample": sample_datas[i]
            })
        return steel
