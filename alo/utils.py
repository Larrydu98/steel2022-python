import re
import psycopg2
import pandas as pd

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


def CountGoodBadNoflag(val):
    '''
    传入的状态标志位是status_fqc
    标签是：flag_lable
    '''
    good_flag = len(val[(val['status_fqc'] == 0) & (val['flag_lable'] == '[1, 1, 1, 1, 1]')])
    bad_flag = len(val[(val['status_fqc'] == 0) & (val['flag_lable'] != '[1, 1, 1, 1, 1]')])
    no_flag = len(val[val['status_fqc'] == 1])

    return good_flag, bad_flag, no_flag


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
