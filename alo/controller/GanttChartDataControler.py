from ..models.ganttChartDataDB import getGanttChartDataDB
import pandas as pd
from ..utils import CountGoodBadNoflag


class getGanttChartData():
    def __init__(self, start_time, end_time):
        rows, col_names = getGanttChartDataDB.ganttChartDataDB(start_time, end_time)
        self.gantt_data = pd.DataFrame(data=rows, columns=col_names).dropna(axis=0, how='all').reset_index(drop=True)
        # print('================')

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
                result[index]['category'].append({
                    'merge_flag': val['merge_flag'],
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
                        'platetype': df_row['platetype'],
                        'tgtwidth': df_row['tgtwidth'],
                        'tgtlength': df_row['tgtlength'],
                        'tgtthickness': df_row['tgtthickness'],
                        'flag_lable': self.JudgePlateQuality(df_row)
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
        merge_limit = 6
        # 判断合并冲突的最大极限
        merrge_conflict = 4
        res = []
        for batch_index, batch_val in enumerate(batch_plate):
            # 1是不能合并，0是可以合并
            category_plate = []
            might_merge_index = batch_val['platetype'].value_counts()[
                batch_val['platetype'].value_counts() >= merge_limit].index.tolist()
            cannot_merge_index = batch_val['platetype'].value_counts()[
                batch_val['platetype'].value_counts() < merge_limit].index.tolist()
            for index, value in enumerate(cannot_merge_index):
                category_plate.append({'merge_flag': False, 'data': batch_val.loc[batch_val['platetype'] == value]})
            for i, val in enumerate(might_merge_index):
                might_merge_arr_index = batch_val.loc[batch_val['platetype'] == val].index
                result = self.JudgeMerge(might_merge_arr_index)
                for res_index, res_val in enumerate(result):
                    # print(i, res_index)
                    if len(res_val) >= merge_limit:
                        category_plate.append({'merge_flag': True, 'data': batch_val.loc[res_val]})
                    else:
                        category_plate.append({'merge_flag': False, 'data': batch_val.loc[res_val]})
                    category_plate.sort(key=lambda k: (k.get('data').iloc[0].toc))
                    # sorted(category_plate, key=lambda x: x.iloc[0].toc)
            res.append(category_plate)
        # for index, value in enumerate(batch_plate):
        #     category_value_plate = []
        #     category_name = value['platetype'].unique()
        #     for i, val in enumerate(category_name):
        #         category_value_plate.append(value.loc[value['platetype'] == val])
        #     category_plate.append(category_value_plate)
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
                if val - data[index - 1] > 3:
                    interval_list.append(data[index_location: index])
                    index_location = index
        interval_list.append(data[index_location:])
        return interval_list
