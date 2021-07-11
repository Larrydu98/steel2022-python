'''
TemperatureP12456Pre
'''
# https://scikit-learn.org/stable/modules/svm.html#svr
from flask import json
import pandas as pd
import numpy as np
import re
from scipy import interpolate

class TemperatureP12456Pre:
    '''
    TemperatureP12456Pre
    '''
    def __init__(self):
        self.paramsIllegal = False
        try:
            print('这个方法暂时没有参数')
        except Exception:
            self.paramsIllegal = True

    def getorder(self,exitscanner_data_new_temp, PostiIntervalTS=0.1):
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
        exitscanner_data_new_finish_temp1 = exitscanner_data_new_finish.sort_values(
            by='position', axis=0, ascending=True)

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

    def getp1p2p4p5p6(self,raw_data, number=100, StratPostiThreshold=0.5, EndPostiThreshold=0.5,
                        PostiIntervalTS=0.1):

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

        all_measured_data_df = pd.DataFrame(data=all_measured_data_list, columns=column).astype(
            'float64')
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
        exitscanner_data_new_finish = exitscanner_data_new_temp[
            (Position_i_i1.Position_i_1 - Position_i_i1.position) > PostiIntervalTS]
        ccp1p6_data = exitscanner_data_new_finish

        p1_temp = ccp1p6_data[
            (ccp1p6_data['position'] >= (p1_temp_position - StratPostiThreshold)) & (
                        ccp1p6_data['position'] <= (
                            plate_length + p1_temp_position + EndPostiThreshold)) & (
                        ccp1p6_data.TempP1 > cc_temp_lowerlimit1)][['position', 'TempP1']]
        p2L_temp = ccp1p6_data[
            (ccp1p6_data['position'] >= (p2L_temp_position - StratPostiThreshold)) & (
                        ccp1p6_data['position'] <= (
                            plate_length + p2L_temp_position + EndPostiThreshold)) & (
                        ccp1p6_data.TempP2L > cc_temp_lowerlimit2L)][['position', 'TempP2L']]
        p4_temp = ccp1p6_data[
            (ccp1p6_data['position'] >= (p4_temp_position - StratPostiThreshold)) & (
                        ccp1p6_data['position'] <= (
                            plate_length + p4_temp_position + EndPostiThreshold)) & (
                        ccp1p6_data.TempP4 > cc_temp_lowerlimit4)][['position', 'TempP4']]
        p5_temp = ccp1p6_data[
            (ccp1p6_data['position'] >= (p5_temp_position - StratPostiThreshold)) & (
                        ccp1p6_data['position'] <= (
                            plate_length + p5_temp_position + EndPostiThreshold)) & (
                        ccp1p6_data.TempP5 > cc_temp_lowerlimit5)][['position', 'TempP5']]
        p6_temp = ccp1p6_data[(ccp1p6_data['position'] >= (p6_temp_position - StratPostiThreshold)) & (ccp1p6_data['position'] <= (
                            plate_length + p6_temp_position + EndPostiThreshold)) & (ccp1p6_data.TempP6 > cc_temp_lowerlimit6)][['position', 'TempP6']]

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

        cc_temp_cleaned_feature = ['TempP1', 'TempP2L', 'TempP4', 'TempP5', 'TempP6']

        cc_temp_cleaned = pd.DataFrame(np.array([p1_temp_y, p2L_temp_y, p4_temp_y, p5_temp_y, p6_temp_y]).T,
            columns=cc_temp_cleaned_feature)

        return cc_temp_cleaned

    def general_call(self, custom_input):
        '''
        general_call
        '''

        # 对P12456数据进行预处理
        number = 100

        p1p2p4p5p6_temp_all_np = []

        for i in range(len(custom_input)):
            print(i)
            raw_data = str(custom_input.content[i])
            Scanner_temp_cleaned = self.getp1p2p4p5p6(raw_data,number)
            p1p2p4p5p6_temp_np = Scanner_temp_cleaned.T.values.reshape(1, 5 * number)

            p1p2p4p5p6_temp_all_np.append(p1p2p4p5p6_temp_np)

        p1p2p4p5p6_temp_all_np = np.array(p1p2p4p5p6_temp_all_np).reshape(len(p1p2p4p5p6_temp_all_np), 5 * number)
        p1p2p4p5p6_temp_all_df_noupid = pd.DataFrame(p1p2p4p5p6_temp_all_np)
        p1p2p4p5p6_temp_all_df = pd.concat([custom_input.upid,p1p2p4p5p6_temp_all_df_noupid], axis=1)

        # 数据清洗
        p1p2p4p5p6_temp_all_df_drop = p1p2p4p5p6_temp_all_df.dropna(axis=0, how='any')
        p1p2p4p5p6_temp_all_df_drop = p1p2p4p5p6_temp_all_df_drop.drop_duplicates('upid', 'last')
        p1p2p4p5p6_temp_all_df_drop = p1p2p4p5p6_temp_all_df_drop.reset_index()
        p1p2p4p5p6_temp_all_df_drop = p1p2p4p5p6_temp_all_df_drop.drop(['index'], axis=1)


        # ####### 转变为模型的输入（deep5）
        # data_x_deep5 = p1p2p4p5p6_temp_all_df_drop.drop(['upid'], axis=1)

        # data_x_deep5_np = ((data_x_deep5 - data_x_deep5.min()) / (data_x_deep5.max() - data_x_deep5.min())).values

        # # 集成成参数
        # test_size = 0.1
        # x_train_deep5, x_test_deep5 = train_test_split(data_x_deep5_np, test_size=test_size, random_state=42)

        # x_train_num = int(len(data_x_deep5) * (1 - test_size))
        # x_test_num = int(len(data_x_deep5) * test_size) + 1

        # x_train_deep5 = x_train_deep5.reshape((x_train_num, 5, 100))
        # x_test_deep5 = x_test_deep5.reshape((x_test_num, 5, 100))

        # # Data Preperation

        # img_H_deep5 = x_train_deep5.shape[1]
        # img_W_deep5 = x_train_deep5.shape[2]
        # depth_deep5 = 1

        # x_train_deep5 = x_train_deep5.astype('float32')
        # x_test_deep5 = x_test_deep5.astype('float32')

        # x_train_deep5 = x_train_deep5.reshape(x_train_deep5.shape[0], img_H_deep5, img_W_deep5,
        #                                         depth_deep5)
        # x_test_deep5 = x_test_deep5.reshape(x_test_deep5.shape[0], img_H_deep5, img_W_deep5,
        #                                     depth_deep5)

        #     ##### 输出x_train_wide，x_test_wide，y_train，y_test
        CcContentData = p1p2p4p5p6_temp_all_df_drop

        return CcContentData
       





