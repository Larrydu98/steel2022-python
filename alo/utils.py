import re
import psycopg2
import pandas as pd

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
