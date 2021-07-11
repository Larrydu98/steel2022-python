'''
Temperature2DPre
'''
# https://scikit-learn.org/stable/modules/svm.html#svr
from flask import json
import pandas as pd
import numpy as np
import re
from scipy import interpolate

class Temperature2DPre:
    '''
    Temperature2DPre
    '''
    def __init__(self):
        self.paramsIllegal = False
        try:
            print('这个方法暂时没有参数')
        except Exception:
            self.paramsIllegal = True

    def getCctemperature2D(self,raw_data, number=100, StartStepThreshold=300, StratPostiThreshold=0,
                EndPostiThreshold=0.5, PostiIntervalTS=0.1):

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

        plate_length = float(length_data[4])

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

        all_measured_data_df = pd.DataFrame(data=all_measured_data_list, columns=column).astype(
            'float64')
        # all_measured_data_df.set_index(["Posit."], inplace=True)
        all_measured_data_df = all_measured_data_df.dropna(axis=0, how='any')
        exitscanner_feature = ['Posit.', 'Scanner00', 'Scanner01', 'Scanner02', 'Scanner03',
                                'Scanner04', 'Scanner05', 'Scanner06', 'Scanner07', 'Scanner08']
        # entryscanner_feature = [ 'EntScan00', 'EntScan01', 'EntScan02', 'EntScan03', 'EntScan04', 'EntScan05', 'EntScan06', 'EntScan07', 'EntScan08']
        exitscanner_data = all_measured_data_df[exitscanner_feature]
        # entryscanner_data = all_measured_data_df[entryscanner_feature]

        x = ['Position', 'Scanner00', 'Scanner01', 'Scanner02', 'Scanner03', 'Scanner04',
                'Scanner05', 'Scanner06', 'Scanner07', 'Scanner08']
        exitscanner_data.columns = x

        start_flag = 0
        end_flag = len(exitscanner_data)

        for i in range(len(exitscanner_data) - 1):
            if ((exitscanner_data.iloc[i + 1] - exitscanner_data.iloc[
                i]).sum() > StartStepThreshold):
                if start_flag == 0:
                    start_flag = i

        start_position = exitscanner_data.Position[start_flag] - StratPostiThreshold
        end_position = exitscanner_data.Position[
                            start_flag] + plate_length / 1000 + EndPostiThreshold

        exitscanner_data_new = exitscanner_data[
            (exitscanner_data['Position'] >= start_position) & (
                        exitscanner_data['Position'] <= end_position)]

        exitscanner_data_new['Position'] = exitscanner_data_new['Position'].apply(
            lambda x: x - exitscanner_data_new['Position'].min())
        exitscanner_data_new = exitscanner_data_new.drop_duplicates(subset=['Position'],
                                                                    keep='last')

        # 数据清洗
        exitscanner_data_new = exitscanner_data_new.sort_values(by='Position', axis=0,
                                                                ascending=True)

        exitscanner_data_new_temp = exitscanner_data_new
        exitscanner_data_new_temp = exitscanner_data_new_temp.reset_index()
        exitscanner_data_new_temp = exitscanner_data_new_temp.drop(['index'], axis=1)

        Position_i = pd.DataFrame(exitscanner_data_new_temp.Position)
        Position_i_1 = exitscanner_data_new_temp.Position.drop(0, axis=0)
        Position_i_1_np = np.insert(Position_i_1.values, len(Position_i_1.values), values=0)
        Position_i_1 = pd.DataFrame(Position_i_1_np, columns=['Position_i_1'])
        Position_i_i1 = pd.concat([Position_i_1, Position_i], axis=1)
        exitscanner_data_new_finish = exitscanner_data_new_temp[
            (Position_i_i1.Position_i_1 - Position_i_i1.Position) > PostiIntervalTS]

        tck_Scanner00_temp = interpolate.splrep(exitscanner_data_new_finish.Position,
                                                exitscanner_data_new_finish.Scanner00)
        Scanner00_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner00_y = interpolate.splev(Scanner00_x, tck_Scanner00_temp)

        tck_Scanner01_temp = interpolate.splrep(exitscanner_data_new_finish.Position,
                                                exitscanner_data_new_finish.Scanner01)
        Scanner01_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner01_y = interpolate.splev(Scanner01_x, tck_Scanner01_temp)

        tck_Scanner02_temp = interpolate.splrep(exitscanner_data_new_finish.Position,
                                                exitscanner_data_new_finish.Scanner02)
        Scanner02_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner02_y = interpolate.splev(Scanner02_x, tck_Scanner02_temp)

        tck_Scanner03_temp = interpolate.splrep(exitscanner_data_new_finish.Position,
                                                exitscanner_data_new_finish.Scanner03)
        Scanner03_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner03_y = interpolate.splev(Scanner03_x, tck_Scanner03_temp)

        tck_Scanner04_temp = interpolate.splrep(exitscanner_data_new_finish.Position,
                                                exitscanner_data_new_finish.Scanner04)
        Scanner04_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner04_y = interpolate.splev(Scanner04_x, tck_Scanner04_temp)

        tck_Scanner05_temp = interpolate.splrep(exitscanner_data_new_finish.Position,
                                                exitscanner_data_new_finish.Scanner05)
        Scanner05_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner05_y = interpolate.splev(Scanner05_x, tck_Scanner05_temp)

        tck_Scanner06_temp = interpolate.splrep(exitscanner_data_new_finish.Position,
                                                exitscanner_data_new_finish.Scanner06)
        Scanner06_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner06_y = interpolate.splev(Scanner06_x, tck_Scanner06_temp)

        tck_Scanner07_temp = interpolate.splrep(exitscanner_data_new_finish.Position,
                                                exitscanner_data_new_finish.Scanner07)
        Scanner07_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner07_y = interpolate.splev(Scanner07_x, tck_Scanner07_temp)

        tck_Scanner08_temp = interpolate.splrep(exitscanner_data_new_finish.Position,
                                                exitscanner_data_new_finish.Scanner08)
        Scanner08_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner08_y = interpolate.splev(Scanner08_x, tck_Scanner08_temp)

        Scanner_temp_cleaned_feature = ['Scanner00', 'Scanner01', 'Scanner02', 'Scanner03', 'Scanner04', 'Scanner05', 
                                        'Scanner06', 'Scanner07', 'Scanner08']

        Scanner_temp_cleaned = pd.DataFrame(np.array([Scanner00_y, Scanner01_y, Scanner02_y, Scanner03_y, Scanner04_y, Scanner05_y,
                                                        Scanner06_y, Scanner07_y, Scanner08_y]).T, columns=Scanner_temp_cleaned_feature)

        #     Scanner_temp_cleaned[Scanner_temp_cleaned>1000]=600

        return Scanner_temp_cleaned


    def general_call(self, custom_input):
        '''
        general_call
        '''
        # 对二维温度场数据进行处理
        number = 100
        Scanner_temp_all_np = []

        for i in range(len(custom_input)):
            print(i)
            raw_data = str(custom_input.content[i])
            Scanner_temp_cleaned = self.getCctemperature2D(raw_data, number)
            Scanner_temp_np = Scanner_temp_cleaned.T.values.reshape(1, number * 9)

            Scanner_temp_all_np.append(Scanner_temp_np)

        Scanner_temp_all_np = np.array(Scanner_temp_all_np).reshape(len(Scanner_temp_all_np),number * 9)
        Scanner_temp_all_df_noupid = pd.DataFrame(Scanner_temp_all_np)
        Scanner_temp_all_df = pd.concat([custom_input.upid,Scanner_temp_all_df_noupid], axis=1)

        Scanner_temp_all_df_drop = Scanner_temp_all_df.dropna(axis=0, how='any')
        Scanner_temp_all_df_drop = Scanner_temp_all_df_drop.drop_duplicates('upid', 'last')
        Scanner_temp_all_df_drop = Scanner_temp_all_df_drop.reset_index()
        Scanner_temp_all_df_drop = Scanner_temp_all_df_drop.drop(['index'], axis=1)



        # ####### 转变为模型的输入（deep4）
        # data_x_deep4 = Scanner_temp_all_df_drop.drop(['upid'], axis=1)

        # data_x_deep4_np = ((data_x_deep4 - data_x_deep4.min()) / (data_x_deep4.max() - data_x_deep4.min())).values

        # # 集成成参数
        # test_size = 0.1
        # x_train_deep4, x_test_deep4 = train_test_split(data_x_deep4_np, test_size=test_size, random_state=42)

        # x_train_num = int(len(data_x_deep4) * (1 - test_size))
        # x_test_num = int(len(data_x_deep4) * test_size) + 1

        # x_train_deep4 = x_train_deep4.reshape((x_train_num, 9, 100))
        # x_test_deep4 = x_test_deep4.reshape((x_test_num, 9, 100))

        # # Data Preperation

        # img_H_deep4 = x_train_deep4.shape[1]
        # img_W_deep4 = x_train_deep4.shape[2]
        # depth_deep4 = 1

        # x_train_deep4 = x_train_deep4.astype('float32')
        # x_test_deep4 = x_test_deep4.astype('float32')

        # x_train_deep4 = x_train_deep4.reshape(x_train_deep4.shape[0], img_H_deep4, img_W_deep4,
        #                                       depth_deep4)
        # x_test_deep4 = x_test_deep4.reshape(x_test_deep4.shape[0], img_H_deep4, img_W_deep4,
        #                                     depth_deep4)

        # ##### 输出x_train_wide，x_test_wide，y_train，y_test
        CcContentData = Scanner_temp_all_df_drop
        return CcContentData
       





