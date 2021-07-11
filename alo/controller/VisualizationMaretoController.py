'''
modelTransferController
'''

import math
import json
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import groupby
from scipy import interpolate
from dateutil.parser import parse


class getMaretoDataController:
    '''
    getSixDpictureUpDownQuantileController
    '''

    def __init__(self):
        pass
        # print('生成实例')

    def run(self, startTime,endTime):
        data_api = 'http://java-serve:8080/web-ssm/myf/RollingTimeVisualizationMaretoController/selectRollingTimeVisualizationMaretoDataDto/'+startTime+'/'+endTime+'/0/5/all/all/10/10/10/100/all/60/'

        mareychart_data = requests.get(data_api)
        mareychart_data = pd.DataFrame(json.loads(mareychart_data.content))

        return mareychart_data


class getSixDpictureUpDownQuantileController:
    '''
    getSixDpictureUpDownQuantileController
    '''

    def __init__(self):
        pass
        # print('生成实例')

    def run(self,MaretoData,startUpid,endUpid,UpQuantile=0.9,DownQuantile=0.1):

        mareychart_data_byUpDown = MaretoData[(MaretoData['upid'] > startUpid) & (MaretoData['upid'] < endUpid)][['ave_temp_dis','charging_temp_act','crowntotal','finishtemptotal',
                                                                                                                  'slab_length','slab_thickness','slab_weight_act','slab_width']]

        count_df = mareychart_data_byUpDown.describe(percentiles=[DownQuantile, UpQuantile])

        DownQuantileData = count_df.iloc[[4]]
        UpQuantileData = count_df.iloc[[6]]


        return UpQuantileData,DownQuantileData

class getDataMaretoStationDataController:
    '''
    getDataMaretoStationDataController
    '''

    def __init__(self):
        pass
        # print('生成实例')

    def run(self, startTime,endTime):
        data_api = 'http://java-serve:8080/web-ssm/myf/RollingTimeVisualizationMaretoController/selectRollingTimeVisualizationMaretoStationDto/'+startTime+'/'+endTime+'/0/5/all/all/10/10/10/10/all/'

        station_data = requests.get(data_api)
        station_data = pd.DataFrame(json.loads(station_data.content))


        return station_data


class MaretoLineMergeController:
    '''
    MaretoLineMergeController
    '''

    def __init__(self):
        pass
        # print('生成实例')

    def run(self, mareychart_data,station_data,lineCrude=1,lineCrudeIncrease=2):

        upids = mareychart_data.index.values
        slabids = mareychart_data['slabid'].values
        stations = station_data['name'].values
        distances = station_data['distance'].values
        stations_df = pd.DataFrame(stations)

        Initial_Time = pd.DataFrame(mareychart_data.stops[0]).time[0]
        new_mareychart_data = pd.DataFrame(data=np.zeros((len(upids), len(stations))),columns=stations, index=upids)

        for i in range(len(upids)):

            stations_temp = []
            plate_data = pd.DataFrame(mareychart_data['stops'][upids[i]])

            for k in range(len(plate_data)):
                station_name = \
                pd.DataFrame.from_dict(plate_data['station'][k], orient='index').T['name'][0]
                stations_temp.append(station_name)

            stations_temp_df = pd.DataFrame(np.array(stations_temp))

            for j in range(len(stations)):

                if (stations_temp_df[stations_temp_df == stations[j]].dropna().empty):
                    timestemp = 0
                #             print(timess)
                else:
                    station_index = \
                    stations_temp_df[stations_temp_df == stations[j]].dropna().index.values[0]
                    time = plate_data['time'][station_index]
                    time_span = (parse(time) - parse(Initial_Time)).total_seconds()
                    #             print(time_span)
                    new_mareychart_data[stations[j]][upids[i]] = time_span

        new_mareychart_data.columns = distances
        new_mareychart_data.iloc[[0], [0]] = math.exp(-45.17)
        for i in range(len(upids)):
            station_data = new_mareychart_data.iloc[i][new_mareychart_data.iloc[i] != 0]
            if (len(station_data) != len(stations)):
                x = station_data.index.values
                y = station_data.values
                yinterp = np.interp(distances, x, y)
                new_mareychart_data.iloc[i] = yinterp
        #         print(len(station_data))
        #         print(yinterp)

        new_mareychart_data.columns = stations

        first_row = pd.DataFrame(data=np.zeros((1, len(stations))), columns=stations)
        new_mareychart_timespan_data_temp = pd.concat([new_mareychart_data, first_row]).reset_index().astype('float') - pd.concat([first_row, new_mareychart_data]).reset_index().astype('float')
        new_mareychart_timespan_data = (new_mareychart_timespan_data_temp.drop([0, len(new_mareychart_timespan_data_temp) - 1])).drop(['index'], axis=1)
        timespan_avg = new_mareychart_timespan_data.mean(axis=1)
        timespan_std = new_mareychart_timespan_data.std(axis=1)
        # timespan_skew = new_mareychart_timespan_data.skew(axis = 1)
        timespan_index = new_mareychart_timespan_data[(timespan_std <= 30) & (timespan_avg <= 300)].index.values

        timespan = []
        timespan_list_temp = []

        fun = lambda x: x[1] - x[0]
        for k, g in groupby(enumerate(timespan_index), fun):
            l1 = [j for i, j in g]
            if len(l1) >= 1:
                timespan = [min(l1), max(l1)]
                timespan_list_temp.append(timespan)
            else:
                timespan_list_temp.append(timespan)

        timespan_df = pd.DataFrame(timespan_list_temp)
        timespan_df.columns = ['start', 'end']
        timespan_df.start = timespan_df.start - 1
        timespan_df['start_upid'] = np.zeros([len(timespan_df)])
        timespan_df['end_upid'] = np.zeros([len(timespan_df)])
        timespan_df['start_time'] = np.zeros([len(timespan_df)])
        timespan_df['end_time'] = np.zeros([len(timespan_df)])

        timespan_df['start_upid'] = timespan_df['start_upid'].apply(str)
        timespan_df['end_upid'] = timespan_df['end_upid'].apply(str)
        timespan_df['start_time'] = timespan_df['start_time'].apply(str)
        timespan_df['end_time'] = timespan_df['end_time'].apply(str)

        for i in range(timespan_df.shape[0]):
            start_index = timespan_df['start'][i]
            end_index = timespan_df['end'][i]

            timespan_df['start_upid'][i] = mareychart_data.index[start_index]
            timespan_df['end_upid'][i] = mareychart_data.index[end_index]
            timespan_df['start_time'][i] = pd.DataFrame(mareychart_data.stops[start_index]).time[0]
            timespan_df['end_time'][i] = pd.DataFrame(mareychart_data.stops[end_index]).time[0]

        mareychart_data_upid = mareychart_data.reset_index()
        mareychart_data_upid['lineCrude'] = lineCrude
        mareychart_data_upid['upid_after'] = mareychart_data_upid.upid

        for i in range(len(timespan_df)):

            start_upid = timespan_df.start_upid[i]
            end_upid = timespan_df.end_upid[i]

            start = timespan_df.start[i]
            end = timespan_df.end[i]

            mareychart_merge_data = mareychart_data_upid[(mareychart_data_upid.upid >= start_upid) & (mareychart_data_upid.upid <= end_upid)]
            # print(i)

            if start == 0:
                merge_after_data = mareychart_merge_data.iloc[[int((end - start) / 2)]]
                merge_after_data.lineCrude = lineCrudeIncrease * (end - start)
                merge_after_data.upid_after.iloc[0] = mareychart_merge_data.upid.tolist()

                mareychart_data_behind = mareychart_data_upid.loc[end + 1:]
                mareychart_data_upid = merge_after_data.append(mareychart_data_behind)

            else:
                mareychart_data_front = mareychart_data_upid.loc[0:start - 1]

                merge_after_data = mareychart_merge_data.iloc[[int((end - start) / 2)]]
                merge_after_data.lineCrude = lineCrudeIncrease * (end - start)
                merge_after_data.upid_after.iloc[0] = mareychart_merge_data.upid.tolist()

                mareychart_data_behind = mareychart_data_upid.loc[end + 1:]

                mareychart_data_upid = mareychart_data_front.append(merge_after_data).append(mareychart_data_behind)

        mareychart_data_upid = mareychart_data_upid.reset_index().drop(['index'], axis=1)

        return mareychart_data_upid


# instance = modelTransferController('2018-09-01 00:00:00', '2018-09-02 00:00:00')
# instance.run()