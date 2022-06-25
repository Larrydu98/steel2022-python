import re
import psycopg2
import pandas as pd
import numpy as np
import datetime as dt

single_dimensional_variable = ["charging_temp_act", "tgtplatelength2", "tgtplatethickness2", "tgtwidth", "slab_length",
                               "slab_thickness", "slab_weight_act", "slab_width",
                               "ave_temp_1", "ave_temp_2", "ave_temp_dis", "ave_temp_pre", "ave_temp_soak",
                               "ave_temp_entry_1", "ave_temp_entry_2", "ave_temp_entry_pre",
                               "ave_temp_entry_soak", "center_temp_dis", "center_temp_entry_1", "center_temp_entry_2",
                               "center_temp_entry_pre", "center_temp_entry_soak",
                               "temp_uniformity_dis", "temp_uniformity_entry_1", "temp_uniformity_entry_2",
                               "temp_uniformity_entry_pre", "temp_uniformity_entry_soak",
                               "skid_temp_dis", "skid_temp_entry_1", "skid_temp_entry_2", "skid_temp_entry_pre",
                               "skid_temp_entry_soak", "staying_time_1", "staying_time_2",
                               "staying_time_pre", "staying_time_soak", "sur_temp_dis", "sur_temp_entry_1",
                               "sur_temp_entry_2", "sur_temp_entry_pre", "sur_temp_entry_soak",
                               "meas_temp_0", "meas_temp_1", "meas_temp_10", "meas_temp_11", "meas_temp_12",
                               "meas_temp_13", "meas_temp_14", "meas_temp_15", "meas_temp_16",
                               "meas_temp_17", "meas_temp_18", "meas_temp_19", "meas_temp_2", "meas_temp_3",
                               "meas_temp_4", "meas_temp_5", "meas_temp_6", "meas_temp_7",
                               "meas_temp_8", "meas_temp_9", "t_0", "t_1", "t_2", "t_3", "t_4", "t_5", "t_6", "pass",
                               "botbrplatecountfm", "botbrplatecountrm",
                               "botwrplatecountfm", "botwrplatecountrm", "crownbody", "crownhead", "crowntail",
                               "crowntotal", "devcrownbody", "devcrownhead", "devcrowntail",
                               "devcrowntotal", "devfinishtempbody", "devfinishtemphead", "devfinishtemptail",
                               "devfinishtemptotal", "wedgebody", "wedgehead", "wedgetail",
                               "wedgetotal", "devwedgebody", "devwedgehead", "devwedgetail", "devwedgetotal",
                               "finishtempbody", "finishtemphead", "finishtemptail",
                               "finishtemptotal", "avg_fct", "avg_p1", "avg_p2", "avg_p5", "avg_sct", "max_fct",
                               "max_p1", "max_p2", "max_p5", "max_sct",
                               "min_fct", "min_p1", "min_p2", "min_p5", "min_sct", "std_fct", "std_p1", "std_p2",
                               "std_p5", "std_sct"]
without_cooling_single_dimensional_variable = ["charging_temp_act", "tgtplatelength2", "tgtplatethickness2", "tgtwidth",
                                               "slab_length", "slab_thickness", "slab_weight_act", "slab_width",
                                               "ave_temp_1", "ave_temp_2", "ave_temp_dis", "ave_temp_pre",
                                               "ave_temp_soak", "ave_temp_entry_1", "ave_temp_entry_2",
                                               "ave_temp_entry_pre",
                                               "ave_temp_entry_soak", "center_temp_dis", "center_temp_entry_1",
                                               "center_temp_entry_2", "center_temp_entry_pre", "center_temp_entry_soak",
                                               "temp_uniformity_dis", "temp_uniformity_entry_1",
                                               "temp_uniformity_entry_2", "temp_uniformity_entry_pre",
                                               "temp_uniformity_entry_soak",
                                               "skid_temp_dis", "skid_temp_entry_1", "skid_temp_entry_2",
                                               "skid_temp_entry_pre", "skid_temp_entry_soak", "staying_time_1",
                                               "staying_time_2",
                                               "staying_time_pre", "staying_time_soak", "sur_temp_dis",
                                               "sur_temp_entry_1", "sur_temp_entry_2", "sur_temp_entry_pre",
                                               "sur_temp_entry_soak",
                                               "meas_temp_0", "meas_temp_1", "meas_temp_10", "meas_temp_11",
                                               "meas_temp_12", "meas_temp_13", "meas_temp_14", "meas_temp_15",
                                               "meas_temp_16",
                                               "meas_temp_17", "meas_temp_18", "meas_temp_19", "meas_temp_2",
                                               "meas_temp_3", "meas_temp_4", "meas_temp_5", "meas_temp_6",
                                               "meas_temp_7", "meas_temp_8", "meas_temp_9", "t_0", "t_1", "t_2", "t_3",
                                               "t_4", "t_5", "t_6", "pass", "botbrplatecountfm", "botbrplatecountrm",
                                               "botwrplatecountfm", "botwrplatecountrm", "crownbody", "crownhead",
                                               "crowntail", "crowntotal", "devcrownbody", "devcrownhead",
                                               "devcrowntail",
                                               "devcrowntotal", "devfinishtempbody", "devfinishtemphead",
                                               "devfinishtemptail", "devfinishtemptotal", "wedgebody", "wedgehead",
                                               "wedgetail",
                                               "wedgetotal", "devwedgebody", "devwedgehead", "devwedgetail",
                                               "devwedgetotal", "finishtempbody", "finishtemphead", "finishtemptail",
                                               "finishtemptotal"]
data_names_meas = ["meas_temp_0", "meas_temp_1", "meas_temp_10", "meas_temp_11", "meas_temp_12", "meas_temp_13", "meas_temp_14", "meas_temp_15", "meas_temp_16",
            "meas_temp_17", "meas_temp_18", "meas_temp_19", "meas_temp_2", "meas_temp_3", "meas_temp_4", "meas_temp_5", "meas_temp_6",
            "meas_temp_7", "meas_temp_8", "meas_temp_9"]


def readConfig():
    # f = open('usr/src/config/config.txt')
    # f = open('usr/src/app/config.txt')
    f = open('config.txt')
    for line in f:
        configArr = line.split(' ')
        break
    return configArr

def postArgs(parser):
    # label = ["tgtthickness", "tgtwidth", 'tgtlength', "tgtdischargetemp", "tgttmplatetemp", "cooling"]
    # label = ["tgtthickness", "tgtwidth", 'tgtlength', "tgtdischargetemp", "tgttmplatetemp", "cooling"]
    label = ["tgtthickness", "tgtwidth", 'tgtlength',  "cooling"]

    for index in label:
        parser.add_argument(index, type=str, required=True)
    args = parser.parse_args(strict=True)
    new_args = {}
    cooling = 1
    for arg_index in args:
        if args[arg_index] == '':
            continue
        if arg_index == 'cooling':
            cooling = int(args[arg_index])
            continue
        new_args[arg_index] = float(args[arg_index])
    return new_args, cooling

def allGetSQLData(SQLquery):
    conn = psycopg2.connect(database='BSData20190712', user='liucheng613', password='liucheng613',
                            host='219.216.80.146', port='5432')
    # conn = psycopg2.connect(database='bg', user='postgres', password='woshimima', host='202.118.21.236',port='5432')

    cursor = conn.cursor()  #
    cursor.execute(SQLquery)
    rows = cursor.fetchall()

    # Extract the column names
    col_names = []
    for elt in cursor.description:
        col_names.append(elt[0])

    conn.close()
    return rows, col_names


def countGoodBadNoflag(data, types):
    '''
    types 是传入的一个df还是传入的一个dataframe的单行数据 single
    传入的状态标志位是status_fqc
    标签是：flag_lable
    '''
    if types == 'dataframe':
        good_flag = len(data[(data['status_fqc'] == 0) & (data['flag_lable'] == '[1, 1, 1, 1, 1]')])
        bad_flag = len(data[(data['status_fqc'] == 0) & (data['flag_lable'] != '[1, 1, 1, 1, 1]')])
        no_flag = len(data[data['status_fqc'] == 1])
        return good_flag, bad_flag, no_flag
    elif types == 'single':
        lable = 0
        if data['status_fqc'] == 1:
            label = 404
        elif data['status_fqc'] == 0:
            if str(data['flag_lable']) == '[1, 1, 1, 1, 1]':
                # 1是好板
                lable = 1
            else:
                lable = 0
        return lable




def diagnosisFlag(data, type):
    flag_list = []
    if (type == 'DataFrame'):
        for index, val in data.iterrows():
            if val.status_fqc == 0:
                if val['fqc_label']['method1']['data'] == [1, 1, 1, 1, 1]:
                    flag_list.append(0)
                else:
                    flag_list.append(1)
            else:
                flag_list.append(404)
    return flag_list



class Cluster:
    def __init__(self,all_data, minute_diff, merge_limit, merge_conflict, args, cooling):
        self.all_data = all_data
        self.minute_diff = int(minute_diff)
        self.merge_limit = int(merge_limit)
        self.merge_conflict = int(merge_conflict)
        self.post_table = args
        self.cooling = cooling

    # 获取cluster数据
    def getClusterData(self):
        result = []
        batch_plate = self.batchPlate()
        category_plate = self.categoryPlate(batch_plate)
        return batch_plate, category_plate

    # 批次划分
    def batchPlate(self):
        batch_plate = []
        minute_diff = dt.timedelta(minutes=30)
        pre = 0
        for bat_index, bat_val in self.all_data.iterrows():
            if bat_index == 0:
                continue
            if (bat_val['toc'] - self.all_data.iloc[bat_index - 1, :]['toc']) >= minute_diff:
                batch_plate.append(self.all_data.iloc[pre:bat_index, :])
                pre = bat_index
        batch_plate.append(self.all_data.iloc[pre:, :])
        return batch_plate

    # 划分种类并判断是否可以合并
    def categoryPlate(self, batch_plate):
        res = []
        for batch_index, batch_val in enumerate(batch_plate):
            # 1是不能合并，0是可以合并
            category_plate = []
            might_merge_index = batch_val['platetype'].value_counts()[batch_val['platetype'].value_counts() >= self.merge_limit].index.tolist()
            cannot_merge_index = batch_val['platetype'].value_counts()[batch_val['platetype'].value_counts() < self.merge_limit].index.tolist()
            for index, value in enumerate(cannot_merge_index):
                category_plate.append({'merge_flag': False, 'data': batch_val.loc[batch_val['platetype'] == value]})
            for i, val in enumerate(might_merge_index):
                # print(batch_index, i)
                # 新增长度宽度厚度三个选择
                # if i == 0:
                #     print('debug')
                might_merge_df = batch_val.loc[batch_val['platetype'] == val]
                coding_list, specification_list = self.DataframeLable(might_merge_df)
                # 编码
                might_merge_df.insert(loc=len(might_merge_df.columns), column='coding', value=coding_list)
                groupby_df = might_merge_df.groupby(might_merge_df['coding']).count()
                unable_merge_list = list(groupby_df.drop(groupby_df[groupby_df.upid >= self.merge_limit].index).index)
                might_able_merge_list = list(
                    groupby_df.drop(groupby_df[groupby_df.upid < self.merge_limit].index).index)
                for unable_merge_i in unable_merge_list:
                    category_plate.append({'merge_flag': False, 'data': batch_val.loc[might_merge_df[might_merge_df['coding'] == unable_merge_i].index]})
                # print(batch_index,i)
                for might_able_merge_list_i in might_able_merge_list:
                    result = self.JudgeMerge(might_merge_df[might_merge_df['coding'] == might_able_merge_list_i].index.values.tolist())
                    for res_index, res_val in enumerate(result):
                        # print(i, res_index)
                        if len(res_val) >= self.merge_limit:
                            # category_info = {}
                            # for spe_i, spe_vla in enumerate(specification_list):
                            #     category_info[spe_vla] = specification_list[spe_vla][int(might_merge_df.loc[res_val].iloc[0, :]['coding'][spe_i])]
                            # if self.cooling:
                            #     category_info['cooling'] = int(might_merge_df.loc[res_val].iloc[0, :]['coding'][-1])
                            category_plate.append({'merge_flag': True, 'data': batch_val.loc[res_val]})
                        else:
                            category_plate.append({'merge_flag': False, 'data': batch_val.loc[res_val]})
            category_plate.sort(key=lambda k: (k.get('data').iloc[0].toc))
            for category_plate_index, category_plate_val in enumerate(category_plate):
                category_plate_val['category_index'] = category_plate_index + 1
            res.append(category_plate)
        return res

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

        if self.cooling:
            for coding_index in range(len(coding_list)):
                coding_list[coding_index] += str(data.iloc[coding_index, :]['status_cooling'])


        return coding_list, specification_list

    # 判断是否能够合并
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



class NewCluster:
    def __init__(self,all_data, minute_diff, merge_limit, merge_conflict, args, cooling):
        self.all_data = all_data
        self.minute_diff = int(minute_diff)
        self.merge_limit = int(merge_limit)
        self.merge_conflict = int(merge_conflict)
        self.post_table = args
        self.cooling = cooling

    # 获取cluster数据
    def getClusterData(self):
        result = []
        batch_plate = self.batchPlate()
        category_plate = self.categoryPlate(batch_plate)
        return batch_plate, category_plate

    # 批次划分
    def batchPlate(self):
        batch_plate = []
        minute_diff = dt.timedelta(minutes=30)
        pre = 0
        for bat_index, bat_val in self.all_data.iterrows():
            if bat_index == 0:
                continue
            if (bat_val['toc'] - self.all_data.iloc[bat_index - 1, :]['toc']) >= minute_diff:
                batch_plate.append(self.all_data.iloc[pre:bat_index, :])
                pre = bat_index
        batch_plate.append(self.all_data.iloc[pre:, :])
        return batch_plate

    # 划分种类并判断是否可以合并
    def categoryPlate(self, batch_plate):
        res = []
        for batch_index, batch_val in enumerate(batch_plate):
            # 1是不能合并，0是可以合并
            category_plate = []
            might_merge_index = batch_val['platetype'].value_counts()[batch_val['platetype'].value_counts() >= self.merge_limit].index.tolist()
            cannot_merge_index = batch_val['platetype'].value_counts()[batch_val['platetype'].value_counts() < self.merge_limit].index.tolist()
            for index, value in enumerate(cannot_merge_index):
                category_plate.append({'merge_flag': False, 'data': batch_val.loc[batch_val['platetype'] == value]})
            for i, val in enumerate(might_merge_index):
                # print(batch_index, i)
                # 新增长度宽度厚度三个选择
                # if i == 0:
                #     print('debug')
                might_merge_df = batch_val.loc[batch_val['platetype'] == val]
                coding_list, specification_list = self.DataframeLable(might_merge_df)
                # 编码
                might_merge_df.insert(loc=len(might_merge_df.columns), column='coding', value=coding_list)
                groupby_df = might_merge_df.groupby(might_merge_df['coding']).count()
                unable_merge_list = list(groupby_df.drop(groupby_df[groupby_df.upid >= self.merge_limit].index).index)
                might_able_merge_list = list(
                    groupby_df.drop(groupby_df[groupby_df.upid < self.merge_limit].index).index)
                for unable_merge_i in unable_merge_list:
                    category_plate.append({'merge_flag': False, 'data': batch_val.loc[might_merge_df[might_merge_df['coding'] == unable_merge_i].index]})
                # print(batch_index,i)
                for might_able_merge_list_i in might_able_merge_list:
                    result = self.JudgeMerge(might_merge_df[might_merge_df['coding'] == might_able_merge_list_i].index.values.tolist())
                    for res_index, res_val in enumerate(result):
                        # print(i, res_index)
                        if len(res_val) >= self.merge_limit:
                            resMergeList = []
                            for i in range(len(res_val)):
                                if not resMergeList:
                                    resMergeList.append([res_val[i]])
                                elif res_val[i - 1] + 1 == res_val[i]:
                                    resMergeList[-1].append(res_val[i])
                                else:
                                    resMergeList.append([res_val[i]])
                            dataCluster = []
                            for i in resMergeList:
                                dataCluster.append(batch_val.loc[i])
                            category_plate.append({'merge_flag': True, 'data': batch_val.loc[res_val], 'dataCluster':dataCluster})
                        else:
                            category_plate.append({'merge_flag': False, 'data': batch_val.loc[res_val]})
            category_plate.sort(key=lambda k: (k.get('data').iloc[0].toc))
            for category_plate_index, category_plate_val in enumerate(category_plate):
                category_plate_val['category_index'] = category_plate_index + 1
            res.append(category_plate)
        return res

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

        if self.cooling:
            for coding_index in range(len(coding_list)):
                coding_list[coding_index] += str(data.iloc[coding_index, :]['status_cooling'])


        return coding_list, specification_list

    # 判断是否能够合并
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
