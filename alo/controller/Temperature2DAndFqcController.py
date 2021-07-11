'''
modelTransferController
'''
import pandas as pd
import requests
import json
from .getDataController import getDataController
from .dataFilterController import dataFilterController
from .dataFilterAfter import dataFilterAfter
# from controller.deepCNNSinTestController import deepCNNSinTestController
from .deepCNNTestController import deepCNNTestController
from PIL import Image
import imagehash
import numpy as np
import re
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns

class Temperature2DExtract:
    '''
    Temperature2DExtract
    '''

    def __init__(self):

        print('生成实例')

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

        all_measured_data_df = pd.DataFrame(data=all_measured_data_list, columns=column).astype('float64')
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
            if ((exitscanner_data.iloc[i + 1] - exitscanner_data.iloc[i]).sum() > StartStepThreshold):
                if start_flag == 0:
                    start_flag = i

        start_position = exitscanner_data.Position[start_flag] - StratPostiThreshold
        end_position = exitscanner_data.Position[start_flag] + plate_length / 1000 + EndPostiThreshold

        exitscanner_data_new = exitscanner_data[(exitscanner_data['Position'] >= start_position) & (exitscanner_data['Position'] <= end_position)]

        exitscanner_data_new['Position'] = exitscanner_data_new['Position'].apply(lambda x: x - exitscanner_data_new['Position'].min())
        exitscanner_data_new = exitscanner_data_new.drop_duplicates(subset=['Position'],keep='last')

        # 数据清洗
        exitscanner_data_new = exitscanner_data_new.sort_values(by='Position', axis=0,ascending=True)

        exitscanner_data_new_temp = exitscanner_data_new
        exitscanner_data_new_temp = exitscanner_data_new_temp.reset_index()
        exitscanner_data_new_temp = exitscanner_data_new_temp.drop(['index'], axis=1)

        Position_i = pd.DataFrame(exitscanner_data_new_temp.Position)
        Position_i_1 = exitscanner_data_new_temp.Position.drop(0, axis=0)
        Position_i_1_np = np.insert(Position_i_1.values, len(Position_i_1.values), values=0)
        Position_i_1 = pd.DataFrame(Position_i_1_np, columns=['Position_i_1'])
        Position_i_i1 = pd.concat([Position_i_1, Position_i], axis=1)
        exitscanner_data_new_finish = exitscanner_data_new_temp[(Position_i_i1.Position_i_1 - Position_i_i1.Position) > PostiIntervalTS]

        tck_Scanner00_temp = interpolate.splrep(exitscanner_data_new_finish.Position,exitscanner_data_new_finish.Scanner00)
        Scanner00_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner00_y = interpolate.splev(Scanner00_x, tck_Scanner00_temp)

        tck_Scanner01_temp = interpolate.splrep(exitscanner_data_new_finish.Position,exitscanner_data_new_finish.Scanner01)
        Scanner01_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner01_y = interpolate.splev(Scanner01_x, tck_Scanner01_temp)

        tck_Scanner02_temp = interpolate.splrep(exitscanner_data_new_finish.Position,exitscanner_data_new_finish.Scanner02)
        Scanner02_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner02_y = interpolate.splev(Scanner02_x, tck_Scanner02_temp)

        tck_Scanner03_temp = interpolate.splrep(exitscanner_data_new_finish.Position,exitscanner_data_new_finish.Scanner03)
        Scanner03_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner03_y = interpolate.splev(Scanner03_x, tck_Scanner03_temp)

        tck_Scanner04_temp = interpolate.splrep(exitscanner_data_new_finish.Position,exitscanner_data_new_finish.Scanner04)
        Scanner04_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner04_y = interpolate.splev(Scanner04_x, tck_Scanner04_temp)

        tck_Scanner05_temp = interpolate.splrep(exitscanner_data_new_finish.Position,exitscanner_data_new_finish.Scanner05)
        Scanner05_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner05_y = interpolate.splev(Scanner05_x, tck_Scanner05_temp)

        tck_Scanner06_temp = interpolate.splrep(exitscanner_data_new_finish.Position,exitscanner_data_new_finish.Scanner06)
        Scanner06_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner06_y = interpolate.splev(Scanner06_x, tck_Scanner06_temp)

        tck_Scanner07_temp = interpolate.splrep(exitscanner_data_new_finish.Position,exitscanner_data_new_finish.Scanner07)
        Scanner07_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner07_y = interpolate.splev(Scanner07_x, tck_Scanner07_temp)

        tck_Scanner08_temp = interpolate.splrep(exitscanner_data_new_finish.Position,exitscanner_data_new_finish.Scanner08)
        Scanner08_x = np.linspace(0, exitscanner_data_new_finish.Position.max(), number)
        Scanner08_y = interpolate.splev(Scanner08_x, tck_Scanner08_temp)

        Scanner_temp_cleaned_feature = ['position',
                                        'Scanner00',
                                        'Scanner01',
                                        'Scanner02',
                                        'Scanner03',
                                        'Scanner04',
                                        'Scanner05',
                                        'Scanner06',
                                        'Scanner07',
                                        'Scanner08']

        Scanner_temp_cleaned = pd.DataFrame(np.array([Scanner00_x,
                                                      Scanner00_y,
                                                      Scanner01_y,
                                                      Scanner02_y,
                                                      Scanner03_y,
                                                      Scanner04_y,
                                                      Scanner05_y,
                                                      Scanner06_y,
                                                      Scanner07_y,
                                                      Scanner08_y]).T,
                                            columns=Scanner_temp_cleaned_feature)

        #     Scanner_temp_cleaned[Scanner_temp_cleaned>1000]=600

        return exitscanner_data_new, Scanner_temp_cleaned

    def run(self,raw_data_pd):
        '''
        run
        '''

        number = 100

        raw_data = str(raw_data_pd.content[0])
        exitscanner_data_before, exitscanner_data_after = self.getCctemperature2D(raw_data, number)


        return exitscanner_data_before,exitscanner_data_after

class getTemperature2DQuantileController:
    '''
    getTemperature2DQuantileController
    '''

    def __init__(self):
        print('生成实例')

    def run(self,Temperature2D,UpQuantile=0.9,DownQuantile=0.1):

        count_df = pd.DataFrame(Temperature2D.values.reshape(len(Temperature2D) * len(Temperature2D.iloc[0]), 1)).describe(percentiles=[DownQuantile, UpQuantile])

        DownQuantileData = count_df.iloc[[4]].values[0][0]
        UpQuantileData = count_df.iloc[[6]].values[0][0]


        return UpQuantileData,DownQuantileData


class Temperature2DAndFqcPictureOutputController:
    '''
    Temperature2DAndFqcPictureOutputController
    '''

    def __init__(self):
        print('生成实例')

    def run(self,data,vmin,vmax,fileName):

        f, ax = plt.subplots(figsize=(7, 25))

        sns.set(font_scale=.8)
        sns.set_style({"savefig.dpi": 100})

        ax_exitscanner = sns.heatmap(data, cmap=plt.cm.Reds, cbar=None, vmin=vmin,vmax=vmax)

        fig_pearson = ax_exitscanner.get_figure()

        fig = plt.gcf()
        # fig.set_size_inches(7.0/3,7.0/3)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.axis('off')
        f.savefig("/usr/src/images/"+fileName+".png", format='png', bbox_inches='tight', transparent=True, dpi=300,pad_inches=0)


        return 0


class Temperature2DAndFqcSimilarityController:
    '''
    Temperature2DAndFqcSimilarityController
    '''

    def __init__(self):
        print('生成实例')

    def run(self,FqcFileName,Temp2DFileName):

        fqc_im = Image.open("/usr/src/images/"+FqcFileName+".png")
        temperature2D_im = Image.open("/usr/src/images/"+Temp2DFileName+".png")

        fqc_resized = fqc_im.resize((2000, 7500), Image.BILINEAR)
        temperature2D_resized = temperature2D_im.resize((2000, 7500), Image.BILINEAR)

        hash_size = 16
        mode = 'db4'
        image_scale = 64
        hash1 = imagehash.whash(fqc_resized, image_scale=image_scale, hash_size=hash_size,mode=mode)

        hash2 = imagehash.whash(temperature2D_resized, image_scale=image_scale,hash_size=hash_size, mode=mode)

        Similarity = (1 - (hash1 - hash2) / len(hash1.hash) ** 2)

        return Similarity


# instance = modelTransferController('2018-09-01 00:00:00', '2018-09-02 00:00:00')
# instance.run()