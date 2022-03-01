from ..models.ganttChartDataDB import getGanttChartDataDB
import pandas as pd
from ..utils import CountGoodBadNoflag


class getGanttChartData():
    def __init__(self, start_time, end_time):
        rows, col_names = getGanttChartDataDB.ganttChartDataDB(start_time, end_time)
        self.gantt_data = pd.DataFrame(data=rows, columns=col_names).dropna(axis=0, how='all').reset_index(drop=True)
        print('================')

    def getGanttChartData(self):
        result = []
        batch_plate = self.batchPlate(30)
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

            for i, val in enumerate(value):
                good_flag, bad_flag, no_flag = CountGoodBadNoflag(val)
                # platetype = val.iloc[0,:]['platetype']
                result[index]['category'].append({
                    'platetype': val.iloc[0,:]['platetype'],
                    'good_flag': good_flag,
                    'bad_flag': bad_flag,
                    'no_flag': no_flag,
                    'total': len(val),
                    'startTime': val.iloc[0].toc.strftime("%Y-%m-%d %H:%M:%S"),
                    'endTime': val.iloc[len(val) - 1].toc.strftime("%Y-%m-%d %H:%M:%S"),
                    "tgtwidth_avg": round(val.tgtwidth.mean(), 3),
                    "tgtlength_avg": round(val.tgtlength.mean(), 3),
                    "tgtthickness_avg": round(val.tgtthickness.mean(), 5),
                    "tgtthickness_avg": round(val.tgtthickness.mean(), 5),
                    'detail': []

                })
                # print(val['platetype'])

                for df_index, df_row in val.iterrows():
                    result[index]['category'][i]['detail'].append({
                        'upid': df_row['upid'],
                        'platetype': df_row['platetype'],
                        'tgtwidth': df_row['tgtwidth'],
                        'tgtlength': df_row['tgtlength'],
                        'tgtthickness': df_row['tgtthickness'],
                        'flag_lable': self.JudgePlateQuality(df_row)
                    })

        #
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

    # 划分种类
    def categoryPlate(self, batch_plate):
        category_plate = []
        for index, value in enumerate(batch_plate):
            category_value_plate = []
            category_name = value['platetype'].unique()
            for i, val in enumerate(category_name):
                category_value_plate.append(value.loc[value['platetype'] == val])
            category_plate.append(category_value_plate)
        return category_plate

    def JudgePlateQuality(self, row):
        if (row['status_fqc'] == 0) & (row['flag_lable'] == '[1, 1, 1, 1, 1]'):
            return 0
        elif (row['status_fqc'] == 0) & (row['flag_lable'] != '[1, 1, 1, 1, 1]'):
            return 1
        else:
            return 404
