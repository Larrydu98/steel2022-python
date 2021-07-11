
'''
getDataController
'''
import numpy as np
import pandas as pd
from flask import json
import re
import requests
from flask_restful import Resource, reqparse
import pika, traceback
import os
from ..staticPath import staticPath
from scipy import interpolate


class getDataController:
    '''
    getDataController
    '''
    def __init__(self, dataNumber, startTime, endTime):
        self.dataNumber = int(dataNumber)
        self.startTime = startTime
        self.endTime = endTime

    def run(self):
        path = os.getcwd()
        address = ['http://java-serve:8080/web-ssm/FuMCcTest1/selectFuMCcTest1/2018-9-1%2000%3A00%3A00/2018-9-2%2024%3A00%3A00/all/',
        'http://java-serve:8080/web-ssm/myf/L2Fufladc51/selectFufladc51SamplingBySlabid/all/2018-9-1%2000%3A00%3A00/2018-9-10%2024%3A00%3A00/40/',
        'http://java-serve:8080/web-ssm/myf/M_output_thicknesspg/selectAllthicknesspg/2018-9-1%2000%3A00%3A00/2018-9-10%2024%3A00%3A00/all/',
        'http://java-serve:8080/web-ssm/myf/l2_m_psc_hpass_meas/selectSumPassMHpassMeas/all/5/0/2018-9-1%2000%3A00%3A00/2018-9-10%2024%3A00%3A00/all/',
        'http://java-serve:8080/web-ssm/myf/l2_m_psc_hpass_post_exit/MMMhpassPostExitOutputNewSum/all/2018-9-1%2000%3A00%3A00/2018-9-10%2024%3A00%3A00/',
        'http://java-serve:8080/web-ssm/myf/l2CcExitScanner/selectL2CcExitScannerByPlateid/all/5/0/2018-9-1%2000%3A00%3A00/2018-9-10%2024%3A00%3A00/all/',
        'tag.csv']
        addressSplit = address[self.dataNumber].split("/")
        pattern =  r"(\d{4}-\d{1,2}-\d{1,2})"
        timeNumber = 0    
        for i in range(len(addressSplit)-1):
            result = re.match(r"(\d{4}-\d{1,2}-\d{1,2})", addressSplit[i])
            if result and timeNumber == 0:
                addressSplit[i] = self.startTime
                timeNumber = timeNumber + 1
                continue
            if result and (timeNumber == 1):
                addressSplit[i] = self.endTime
                timeNumber = 0
        symbol = '/'
        myAddress = symbol.join(addressSplit)
        if self.dataNumber != 6:
            dataFromJava = requests.get(myAddress)
            dataFromJava = pd.DataFrame(json.loads(dataFromJava.content))
            # dataFromJava = dataFromJava.to_json(orient='index')
        else:
            data_usid = pd.read_csv(path + staticPath.data_tag_w)
            # data_usid = pd.read_csv(path + staticPath.data_tag_l)
            data_usid_np = data_usid['Unnamed: 0'].values
            Newtag1 = data_usid['Newtag1'].values
            start_flag = 0
            upid = []
            upid_flag_list = []
                
            for i in range(len(data_usid_np)-1):
                if (i== len(data_usid_np)-1)|(data_usid_np[i][0:8] != data_usid_np[i+1][0:8]):
            #         print(i)
                    end_flag = i
                        
                    upid.append(data_usid_np[i][0:8]+"000")
                        
                    usid_num = end_flag-start_flag+1
                    upid_flag=1
                    for j in range(usid_num):
                        if(Newtag1[start_flag+j]==0):
                            upid_flag=0
            #             print(upid_flag)
                    upid_flag_list.append(upid_flag)
                        
                    start_flag = end_flag+1
                    
            upid_df = pd.DataFrame(upid,columns = ['upid'])
            upid_flag_df = pd.DataFrame(upid_flag_list,columns = ['upid_flag'])
            dataFromJava = pd.concat([upid_df, upid_flag_df],axis=1)
        return dataFromJava


class getDataVisualizationMaretoUpidFlagController:
    '''
    getDataVisualizationMaretoUpidFlagController
    '''

    def __init__(self, startTime, endTime):
        self.startTime = startTime
        self.endTime = endTime

    def run(self):
        startTime = self.startTime
        endTime = self.endTime
        api = 'http://java-serve:8080/web-ssm/myf/RollingTimeVisualizationMaretoController/selectRollingTimeVisualizationMaretoDataDto/' + startTime + '/' + endTime + '/0/5/all/all/10/10/10/100/all/60/'
        VisualizationMaretoTime = requests.get(api)
        VisualizationMaretoTime = pd.DataFrame(json.loads(VisualizationMaretoTime.content))

        VisualizationMaretoUpidFlag = VisualizationMaretoTime[['upid','flag']]


        return VisualizationMaretoUpidFlag

class getDataCcContentController:
    '''
    getDataCcContentController
    '''

    def __init__(self, upid):
        self.upid = upid

    def run(self):
        upid = self.upid
        api = 'http://java-serve:8080/web-ssm/myf/l2CcExitScanner/selectL2CcExitScanner/'+upid+'/5/0/2018-9-1%2000%3A00%3A00/2018-9-30%2024%3A00%3A00/all/'
        raw_data_pd = requests.get(api)
        if raw_data_pd.content:
            raw_data_pd = pd.DataFrame(json.loads(raw_data_pd.content))
        else:
            raw_data_pd = '该钢板不存在冷却数据'
        # raw_data_pd = pd.DataFrame(json.loads(raw_data_pd.content))


        return raw_data_pd


class getDataFqcController:
    '''
    getDataCcContentController
    '''

    def __init__(self, upid):
        self.upid = upid

    def run(self):
        upid = self.upid

        dirs = '/usr/src/data/mother_plate_fqc'
        myList = os.listdir(dirs)
        my_dirs_upid_list = []
        for my_dirs in myList:
            my_dirs_upid_list.append(my_dirs[0:11])

        my_dirs_upid_df = pd.DataFrame(my_dirs_upid_list, columns=['upid'])
        # short_data_upid = 'E:\mother_plate_fqc\short_data_upid_csv.csv'

        if (len(my_dirs_upid_df[my_dirs_upid_df.upid.isin([upid])]) != 0):
            mother_plate_fqc_data = pd.read_csv("/usr/src/data/mother_plate_fqc/" + upid + ".csv")
            # fdfafadata = pd.read_csv("/usr/src/data/mother_fqc_before/12.csv")
            # print(fdfafadata)
            # mother_plate_fqc_data = mother_plate_fqc_data.set_index('x2')
        else:
            mother_plate_fqc_data = '该钢板不存在FQC数据'

        dirs_before = '/usr/src/data/mother_fqc_before'
        myList_before = os.listdir(dirs_before)
        my_dirs_upid_list_before = []
        for my_dirs in myList_before:
            my_dirs_upid_list_before.append(my_dirs[0:11])

        my_dirs_upid_before_df = pd.DataFrame(my_dirs_upid_list_before, columns=['upid'])

        if (len(my_dirs_upid_before_df[my_dirs_upid_before_df.upid.isin([upid])]) != 0):
            mother_plate_fqc_before_data = pd.read_csv("/usr/src/data/mother_fqc_before/" + upid + ".csv")
            # fdfafadata = pd.read_csv("/usr/src/data/mother_fqc_before/12.csv")
            # print(fdfafadata)
            # mother_plate_fqc_before_data = mother_plate_fqc_before_data.set_index('Unnamed: 0')
        else:
            mother_plate_fqc_before_data = '该钢板不存在FQC数据'

        return mother_plate_fqc_before_data,mother_plate_fqc_data


class getDataMOutputPgController:
    '''
    getDataMOutputPgController
    '''

    def __init__(self, upid):
        self.upid = upid

    def run(self):
        upid = self.upid

        api = 'http://java-serve:8080/web-ssm/myf/M_output_thicknesspg/selectAllthicknesspg/2018-9-1%2000%3A00%3A00/2018-9-30%2024%3A00%3A00/'+upid+'/'
        M_Pg_Output = requests.get(api)
        M_Pg_Output = pd.DataFrame(json.loads(M_Pg_Output.content))

        centerthickness = np.array(M_Pg_Output['centerthickness'].tolist()[0])
        leftthickness = np.array(M_Pg_Output['leftthickness'].tolist()[0])
        rightthickness = np.array(M_Pg_Output['rightthickness'].tolist()[0])
        position = np.array(M_Pg_Output['position'].tolist()[0])

        if (np.max(centerthickness)>100)|(np.max(leftthickness)>100)|(np.max(rightthickness)>100):
            result = '该钢板的凸度仪数据异常'
        else:
            result = {'position': position.tolist(),
                      'centerthickness': centerthickness.tolist(),
                      'leftthickness': leftthickness.tolist(),
                      'rightthickness': rightthickness.tolist()}

        return result


class getDataPlateThinessPrimaryController:
    '''
    getDataPlateThinessPrimaryController
    '''

    def __init__(self, upid):
        self.upid = upid

    def run(self):
        upid = self.upid

        api = 'http://java-serve:8080/web-ssm/myf/l2GetPlatePrimaryData/selectL2GetPlatePrimaryData/'+upid+'/2018-9-1%2000%3A00%3A00/2018-9-30%2024%3A00%3A00/'
        PlatePrimaryData = requests.get(api)
        PlatePrimaryData = pd.DataFrame(json.loads(PlatePrimaryData.content))

        maxplatethickness2 = PlatePrimaryData.maxplatethickness2.values[0]
        minplatethickness2 = PlatePrimaryData.minplatethickness2.values[0]
        tgtplatethickness2 = PlatePrimaryData.tgtplatethickness2.values[0]


        return {'maxplatethickness2': maxplatethickness2,
                'minplatethickness2': minplatethickness2,
                'tgtplatethickness2': tgtplatethickness2}


class getDataMCcTempController:
    '''
    getDataMCcTempController
    '''

    def __init__(self, upid):
        self.upid = upid

    def run(self):
        upid = self.upid

        Temp_data_api = 'http://java-serve:8080/web-ssm/myf/RollingTimeVisualizationMaretoController/selectRollingTimeVisualizationMaretoTempDto/2018-9-1%2000%3A00%3A00/2018-9-30%2024%3A00%3A00/0/5/'+upid+'/all/40/40/40/40/all/'
        Temp_data = requests.get(Temp_data_api)
        Temp_data = pd.DataFrame(json.loads(Temp_data.content)).T

        time_data_api = 'http://java-serve:8080/web-ssm/myf/RollingTimeVisualizationMaretoController/selectRollingTimeVisualizationMaretoDataDto/2018-9-1%2000%3A00%3A00/2018-9-30%2024%3A00%3A00/0/5/'+upid+'/all/10/10/10/100/all/60/'
        time_data = requests.get(time_data_api)
        time_data = pd.DataFrame(json.loads(time_data.content))

        temp_list = Temp_data.temps.values.tolist()[0]
        time_list = time_data.stops.values.tolist()[0]

        temp_data_list = []
        temp_station_name_list = []

        for i in range(len(temp_list)):
            temp_data_list.append(Temp_data.temps.values.tolist()[0][i]['temp'])
            temp_station_name_list.append(Temp_data.temps.values.tolist()[0][i]['station']['name'])

        temp_data_df = pd.DataFrame(np.array(temp_data_list).reshape(1, -1),columns=temp_station_name_list)

        time_data_list = []
        time_station_name_list = []

        for i in range(len(time_list)):
            if (i != len(time_list) - 1):
                if (time_data.stops.values.tolist()[0][i + 1]['station']['name'] != 'CcStart'):
                    time_data_list.append(time_data.stops.values.tolist()[0][i]['time'])
                    time_station_name_list.append(time_data.stops.values.tolist()[0][i]['station']['name'])
            else:
                time_data_list.append(time_data.stops.values.tolist()[0][i]['time'])
                time_station_name_list.append(time_data.stops.values.tolist()[0][i]['station']['name'])

        time_data_df = pd.DataFrame(np.array(time_data_list).reshape(1, -1),columns=time_station_name_list)
        time_data_df.rename(columns={'CcACCEnd': 'CcEnd'}, inplace=True)

        temp_time_df = pd.concat([temp_data_df, time_data_df], axis=0).T
        temp_time_df.columns = ['Temp', 'Time']
        temp_time_df_drop = temp_time_df.dropna(axis=0, how='any')

        temp_time_df_drop.Time = pd.to_datetime(temp_time_df_drop.Time, format='%Y-%m-%d %H:%M:%S')
        temp_time_df_drop_sortValues = temp_time_df_drop.sort_values(by='Time').iloc[5:len(temp_time_df_drop)]

        temp_time_df_drop_sortValues['Time'] = temp_time_df_drop_sortValues['Time'].apply(lambda x: (x - temp_time_df_drop_sortValues['Time'][0]).seconds)

        return temp_time_df_drop_sortValues


class getDataP12456TempController:
    '''
    getDataP12456TempController
    '''

    def __init__(self, upid):
        self.upid = upid

    def getorder(self, exitscanner_data_new_temp, PostiIntervalTS=0.1):
        exitscanner_data_new_temp = exitscanner_data_new_temp.reset_index()
        exitscanner_data_new_temp = exitscanner_data_new_temp.drop(['index'], axis=1)
        Position_i = pd.DataFrame(exitscanner_data_new_temp.position)
        Position_i_1 = exitscanner_data_new_temp.position.drop(0, axis=0)
        Position_i_1_np = np.insert(Position_i_1.values, len(Position_i_1.values), values=0)
        Position_i_1 = pd.DataFrame(Position_i_1_np, columns=['Position_i_1'])
        Position_i_i1 = pd.concat([Position_i_1, Position_i], axis=1)
        exitscanner_data_new_reverse = exitscanner_data_new_temp[
            (Position_i_i1.Position_i_1 - Position_i_i1.position) < 0]
        if len(exitscanner_data_new_temp[exitscanner_data_new_reverse.index[
                                             len(exitscanner_data_new_reverse) - 2] + 1:]) > 30:
            exitscanner_data_new_finish = exitscanner_data_new_temp[
                                          exitscanner_data_new_reverse.index[
                                              len(exitscanner_data_new_reverse) - 2] + 1:]
        else:
            exitscanner_data_new_finish = exitscanner_data_new_temp
        exitscanner_data_new_finish_temp1 = exitscanner_data_new_finish.sort_values(by='position', axis=0, ascending=True)

        exitscanner_data_new_temp = exitscanner_data_new_finish_temp1
        exitscanner_data_new_temp = exitscanner_data_new_temp.reset_index()
        exitscanner_data_new_temp = exitscanner_data_new_temp.drop(['index'], axis=1)
        Position_i = pd.DataFrame(exitscanner_data_new_temp.position)
        Position_i_1 = exitscanner_data_new_temp.position.drop(0, axis=0)
        Position_i_1_np = np.insert(Position_i_1.values, len(Position_i_1.values), values=0)
        Position_i_1 = pd.DataFrame(Position_i_1_np, columns=['Position_i_1'])
        Position_i_i1 = pd.concat([Position_i_1, Position_i], axis=1)
        exitscanner_data_new_finish = exitscanner_data_new_temp[
            (Position_i_i1.Position_i_1 - Position_i_i1.position) > PostiIntervalTS]
        ccp1p6_data = exitscanner_data_new_finish

        return exitscanner_data_new_finish

    def getp1p2p4p5p6(self, raw_data, number=100, StratPostiThreshold=0.5,EndPostiThreshold=0.5,PostiIntervalTS=0.1):

        p1_temp_position = 12.9
        p2L_temp_position = 51.0
        p4_temp_position = 113
        p5_temp_position = 115.5
        p6_temp_position = 116.3

        cc_temp_lowerlimit1 = 0
        cc_temp_lowerlimit2L = 0
        cc_temp_lowerlimit4 = 0
        cc_temp_lowerlimit5 = 0
        cc_temp_lowerlimit6 = 0

        headnum = raw_data.find('______Output measured values_________________')
        tailnum = raw_data.find('[POSTCALC] ACC Post-Calculation Output:')

        length_headnum = raw_data.find('Length of the plate')
        length_tailnum = raw_data.find('Finishing mill temp.')

        measured_data = raw_data[headnum:tailnum]
        measured_data = measured_data[measured_data.find("No."):]
        measured_datalist = re.split("\\\\n|\\\n|\\n|\n", measured_data)
        measured_datalist[0]

        length_data = raw_data[length_headnum:length_tailnum]
        length_datalist = re.split("\\\\n|\\\n|\\n|\n", length_data)
        length_data = length_datalist[0].split()

        plate_length = float(length_data[4]) / 1000

        column = []
        column_temp = measured_datalist[0].split()
        for data in column_temp:
            if data == 'EmcOSPos9EmcOSPos10':
                column.append('EmcOSPos9')
                column.append('EmcOSPos10')
            elif data == 'EmcDSPos9EmcDSPos10':
                column.append('EmcDSPos9')
                column.append('EmcDSPos10')
            else:
                column.append(data)

        all_measured_data_list = []
        for i in range(1, len(measured_datalist)):
            all_measured_data = measured_datalist[i].split()
            all_measured_data_list.append(all_measured_data)

        all_measured_data_df = pd.DataFrame(data=all_measured_data_list, columns=column).astype('float64')
        # all_measured_data_df.set_index(["Posit."], inplace=True)
        all_measured_data_df = all_measured_data_df.dropna(axis=0, how='any')
        ccp1p6_feature = ['Posit.', 'TempP1', 'TempP2L', 'TempP4', 'TempP5', 'TempP6']
        ccp1p6_data = all_measured_data_df[ccp1p6_feature]

        x = ['position', 'TempP1', 'TempP2L', 'TempP4', 'TempP5', 'TempP6']
        ccp1p6_data.columns = x

        # 去掉position比较相近的点
        exitscanner_data_new_temp = ccp1p6_data
        exitscanner_data_new_temp = exitscanner_data_new_temp.reset_index()
        exitscanner_data_new_temp = exitscanner_data_new_temp.drop(['index'], axis=1)
        Position_i = pd.DataFrame(exitscanner_data_new_temp.position)
        Position_i_1 = exitscanner_data_new_temp.position.drop(0, axis=0)
        Position_i_1_np = np.insert(Position_i_1.values, len(Position_i_1.values), values=0)
        Position_i_1 = pd.DataFrame(Position_i_1_np, columns=['Position_i_1'])
        Position_i_i1 = pd.concat([Position_i_1, Position_i], axis=1)
        exitscanner_data_new_finish = exitscanner_data_new_temp[(Position_i_i1.Position_i_1 - Position_i_i1.position) > PostiIntervalTS]
        ccp1p6_data = exitscanner_data_new_finish

        p1_temp = ccp1p6_data[(ccp1p6_data['position'] >= (p1_temp_position - StratPostiThreshold)) & (
                    ccp1p6_data['position'] <= (
                    plate_length + p1_temp_position + EndPostiThreshold)) & (
                    ccp1p6_data.TempP1 > cc_temp_lowerlimit1)][['position', 'TempP1']]
        p2L_temp = ccp1p6_data[(ccp1p6_data['position'] >= (p2L_temp_position - StratPostiThreshold)) & (
                    ccp1p6_data['position'] <= (
                    plate_length + p2L_temp_position + EndPostiThreshold)) & (
                    ccp1p6_data.TempP2L > cc_temp_lowerlimit2L)][['position', 'TempP2L']]
        p4_temp = ccp1p6_data[(ccp1p6_data['position'] >= (p4_temp_position - StratPostiThreshold)) & (
                    ccp1p6_data['position'] <= (
                    plate_length + p4_temp_position + EndPostiThreshold)) & (
                    ccp1p6_data.TempP4 > cc_temp_lowerlimit4)][['position', 'TempP4']]
        p5_temp = ccp1p6_data[(ccp1p6_data['position'] >= (p5_temp_position - StratPostiThreshold)) & (
                    ccp1p6_data['position'] <= (
                    plate_length + p5_temp_position + EndPostiThreshold)) & (
                    ccp1p6_data.TempP5 > cc_temp_lowerlimit5)][['position', 'TempP5']]
        p6_temp = ccp1p6_data[(ccp1p6_data['position'] >= (p6_temp_position - StratPostiThreshold)) & (
                    ccp1p6_data['position'] <= (
                    plate_length + p6_temp_position + EndPostiThreshold)) & (
                    ccp1p6_data.TempP6 > cc_temp_lowerlimit6)][['position', 'TempP6']]

        p1_temp = self.getorder(p1_temp, PostiIntervalTS)
        p2L_temp = self.getorder(p2L_temp, PostiIntervalTS)
        p4_temp = self.getorder(p4_temp, PostiIntervalTS)
        p5_temp = self.getorder(p5_temp, PostiIntervalTS)
        p6_temp = self.getorder(p6_temp, PostiIntervalTS)

        p1_temp['position'] = p1_temp['position'].apply(lambda x: x - p1_temp['position'].min())
        p2L_temp['position'] = p2L_temp['position'].apply(lambda x: x - p2L_temp['position'].min())
        p4_temp['position'] = p4_temp['position'].apply(lambda x: x - p4_temp['position'].min())
        p5_temp['position'] = p5_temp['position'].apply(lambda x: x - p5_temp['position'].min())
        p6_temp['position'] = p6_temp['position'].apply(lambda x: x - p6_temp['position'].min())

        p1_temp = p1_temp.drop_duplicates(subset=['position'], keep='last')
        p2L_temp = p2L_temp.drop_duplicates(subset=['position'], keep='last')
        p4_temp = p4_temp.drop_duplicates(subset=['position'], keep='last')
        p5_temp = p5_temp.drop_duplicates(subset=['position'], keep='last')
        p6_temp = p6_temp.drop_duplicates(subset=['position'], keep='last')

        # 数据上采样（插值算法）
        tck_p1_temp = interpolate.splrep(p1_temp.position, p1_temp.TempP1)
        p1_temp_x = np.linspace(0, p1_temp.position.max(), number)
        p1_temp_y = interpolate.splev(p1_temp_x, tck_p1_temp)
        # plt.plot(p1_temp_x, p1_temp_y, 'r-')

        tck_p2L_temp = interpolate.splrep(p2L_temp.position, p2L_temp.TempP2L)
        p2L_temp_x = np.linspace(0, p2L_temp.position.max(), number)
        p2L_temp_y = interpolate.splev(p2L_temp_x, tck_p2L_temp)
        # plt.plot(p2L_temp_x, p2L_temp_y, 'r-')

        tck_p4_temp = interpolate.splrep(p4_temp.position, p4_temp.TempP4)
        p4_temp_x = np.linspace(0, p4_temp.position.max(), number)
        p4_temp_y = interpolate.splev(p4_temp_x, tck_p4_temp)
        # plt.plot(p4_temp_x, p4_temp_y, 'r-')

        tck_p5_temp = interpolate.splrep(p5_temp.position, p5_temp.TempP5)
        p5_temp_x = np.linspace(0, p5_temp.position.max(), number)
        p5_temp_y = interpolate.splev(p5_temp_x, tck_p5_temp)
        # plt.plot(p5_temp_x, p5_temp_y, 'r-')

        tck_p6_temp = interpolate.splrep(p6_temp.position, p6_temp.TempP6)
        p6_temp_x = np.linspace(0, p6_temp.position.max(), number)
        p6_temp_y = interpolate.splev(p6_temp_x, tck_p6_temp)
        # plt.plot(p5_temp_x, p5_temp_y, 'r-')

        cc_temp_cleaned_feature = ['postionX', 'TempP1', 'TempP2L', 'TempP4', 'TempP5', 'TempP6']

        cc_temp_cleaned = pd.DataFrame(np.array([p1_temp_x,p1_temp_y, p2L_temp_y, p4_temp_y, p5_temp_y, p6_temp_y]).T,columns=cc_temp_cleaned_feature)

        return cc_temp_cleaned


    def run(self):
        upid = self.upid

        api = 'http://java-serve:8080/web-ssm/myf/l2CcExitScanner/selectL2CcExitScanner/'+upid+'/5/0/2018-9-1%2000%3A00%3A00/2018-9-30%2024%3A00%3A00/all/'
        raw_data_pd = requests.get(api)
        if raw_data_pd.content:
            raw_data_pd = pd.DataFrame(json.loads(raw_data_pd.content))
            P12456Data = self.getp1p2p4p5p6(raw_data_pd.content[0])
        else:
            P12456Data = '该钢板不存在冷却数据'


        return P12456Data


class getDataUpidClassController:
    '''
    getDataUpidClassController
    '''

    def __init__(self, upid):
        self.upid = upid

    def run(self):
        upid = self.upid

        api = 'http://java-serve:8080/web-ssm/FuMCcTest1/selectFuMCcTest1/2018-9-1%2000%3A00%3A00/2018-9-30%2024%3A00%3A00/'+ upid +'/'
        Fu_M_Cc_Output = requests.get(api)
        Fu_M_Cc_Output = pd.DataFrame(json.loads(Fu_M_Cc_Output.content))

        plateBasicInformation = Fu_M_Cc_Output[['upid', 'slab_length', 'slab_width', 'slab_thickness', 'slab_weight_act','productcategory', 'tgtwidth', 'tgtplatelength2', 'tgtplatethickness2','toc']]

        plateBasicInformation.upid.values.tolist()[0]

        return {'母版ID': plateBasicInformation.upid.values.tolist()[0],
                '板坯长度': plateBasicInformation.slab_length.values.tolist()[0]/1000,
                '板坯宽度': plateBasicInformation.slab_width.values.tolist()[0]/1000,
                '板坯厚度': plateBasicInformation.slab_thickness.values.tolist()[0]/1000,
                '板坯重量': plateBasicInformation.slab_weight_act.values.tolist()[0],
                '钢种': plateBasicInformation.productcategory.values.tolist()[0],
                '目标宽度': plateBasicInformation.tgtwidth.values.tolist()[0],
                '目标长度': plateBasicInformation.tgtplatelength2.values.tolist()[0],
                '目标厚度': plateBasicInformation.tgtplatethickness2.values.tolist()[0],
                '生产时间': plateBasicInformation.toc.values.tolist()[0]}
