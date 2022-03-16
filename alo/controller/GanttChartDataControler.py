from ..models.ganttChartDataDB import getGanttChartDataDB
import pandas as pd
from ..utils import CountGoodBadNoflag


class getGanttChartData():
    def __init__(self, parser, start_time, end_time,merge_limit,merge_conflict):
        rows, col_names = getGanttChartDataDB.ganttChartDataDB(start_time, end_time)
        label = ["tgtthickness", "tgtwidth", "tgtlength"]
        for index in label:
            parser.add_argument(index, type=str, required=True)
        self.post_table = parser.parse_args(strict=True)
        self.gantt_data = pd.DataFrame(data=rows, columns=col_names).dropna(axis=0, how='all').reset_index(drop=True)
        self.merge_limit = int(merge_limit)
        self.merge_conflict = int(merge_conflict)

    def getGanttChartData(self):
        # 批次划分的时间差
        minute_diff = 30
        result = []
        batch_plate = self.batchPlate(minute_diff)
        category_plate = self.categoryPlate(batch_plate)
        for index, value in enumerate(category_plate):
            good_flag, bad_flag, no_flag = CountGoodBadNoflag(batch_plate[index])
            result.append({
                'batch': index + 1,
                'good_flag': good_flag,
                'bad_flag': bad_flag,
                'no_flag': no_flag,
                'total': len(batch_plate[index]),
                'startTime': batch_plate[index].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
                'endTime': batch_plate[index].iloc[len(batch_plate[index]) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
                "tgtwidth_avg": round(batch_plate[index].tgtwidth.mean(), 3),
                "tgtlength_avg": round(batch_plate[index].tgtlength.mean(), 3),
                "tgtthickness_avg": round(batch_plate[index].tgtthickness.mean(), 5),
                'category': []
            })
            # print('====================================')
            for i, val in enumerate(value):
                good_flag, bad_flag, no_flag = CountGoodBadNoflag(val['data'])
                # platetype = val.iloc[0,:]['platetype']
                if val['merge_flag']:
                    result[index]['category'].append({
                        'merge_flag': val['merge_flag'],
                        'category_index': val['category_index'],
                        'category_info': val['category_info'],
                        'platetype': val['data'].iloc[0, :]['platetype'],
                        'good_flag': good_flag,
                        'bad_flag': bad_flag,
                        'no_flag': no_flag,
                        'total': len(val['data']),
                        'startTime': val['data'].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
                        'endTime': val['data'].iloc[len(val['data']) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
                        "tgtwidth_avg": round(val['data'].tgtwidth.mean(), 3),
                        "tgtlength_avg": round(val['data'].tgtlength.mean(), 3),
                        "tgtthickness_avg": round(val['data'].tgtthickness.mean(), 5),
                        "tgtthickness_avg": round(val['data'].tgtthickness.mean(), 5),
                        'detail': []

                    })
                else:
                    result[index]['category'].append({
                        'merge_flag': val['merge_flag'],
                        'category_index': val['category_index'],
                        'platetype': val['data'].iloc[0, :]['platetype'],
                        'good_flag': good_flag,
                        'bad_flag': bad_flag,
                        'no_flag': no_flag,
                        'total': len(val['data']),
                        'startTime': val['data'].iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
                        'endTime': val['data'].iloc[len(val['data']) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
                        "tgtwidth_avg": round(val['data'].tgtwidth.mean(), 3),
                        "tgtlength_avg": round(val['data'].tgtlength.mean(), 3),
                        "tgtthickness_avg": round(val['data'].tgtthickness.mean(), 5),
                        "tgtthickness_avg": round(val['data'].tgtthickness.mean(), 5),
                        'detail': []
                    })
                # print(len(val['data']), val['merge_flag'])
                # print(val['platetype'])

                for df_index, df_row in val['data'].iterrows():
                    result[index]['category'][i]['detail'].append({
                        'upid': df_row['upid'],
                        'toc': df_row['toc'].strftime("%Y-%m-%d %H:%M:%S"),
                        'platetype': df_row['platetype'],
                        'tgtwidth': df_row['tgtwidth'],
                        'tgtlength': df_row['tgtlength'],
                        'tgtthickness': df_row['tgtthickness'],
                        'flag_lable': self.JudgePlateQuality(df_row),
                        'status_cooling': df_row['status_cooling']
                    })

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
            cannot_merge_index = batch_val['platetype'].value_counts()[batch_val['platetype'].value_counts() < self.merge_limit].index.tolist()

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
                might_able_merge_list = list(groupby_df.drop(groupby_df[groupby_df.upid < self.merge_limit].index).index)
                for unable_merge_i in unable_merge_list:
                    category_plate.append({'merge_flag': False, 'data': batch_val.loc[might_merge_df[might_merge_df['coding'] == unable_merge_i].index]})

                for might_able_merge_list_i in might_able_merge_list:
                    result = self.JudgeMerge(might_merge_df[might_merge_df['coding'] == might_able_merge_list_i].index.values.tolist())
                    for res_index, res_val in enumerate(result):
                        # print(i, res_index)
                        if len(res_val) >= self.merge_limit:
                            category_info = {}
                            for spe_i, spe_vla in enumerate(specification_list):
                                category_info[spe_vla] = specification_list[spe_vla][int(might_merge_df.loc[res_val].iloc[0,:]['coding'][spe_i])]
                            category_info['cooling'] = int(might_merge_df.loc[res_val].iloc[0,:]['coding'][-1])
                            category_plate.append({'merge_flag': True, 'data': batch_val.loc[res_val], "category_info": category_info})
                        else:
                            category_plate.append({'merge_flag': False, 'data': batch_val.loc[res_val]})
            category_plate.sort(key=lambda k: (k.get('data').iloc[0].toc))
            for category_plate_index,category_plate_val in enumerate(category_plate):
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
    def DataframeLable(self,data):
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
            for index, pd_val in enumerate(pd.cut(data[val],s_bin,labels=False,right=False)):
                coding_list[index] += str(pd_val)

        for coding_index in range(len(coding_list)):
            coding_list[coding_index] += str(data.iloc[coding_index,:]['status_cooling'])

        return coding_list,specification_list







