import re
import psycopg2
import pandas as pd



# selection=[]
# ismissing={'all_processes_statistics_ismissing':True,'cool_ismissing':True,'fu_temperature_ismissing':True,'m_ismissing':True,'fqc_ismissing':True}   
# tgtwidthSelect=['2.895','4.48']
# tgtlengthSelect=['16.137','40.382']
# tgtthicknessSelect=['0.0102','0.0232']
# tocSelect=['2018-12-01 01:41:43','2018-12-10 12:11:43']
# UpidSelect=['18C01025000','18C01032000'];
# Platetypes=['KA36-TM','AB/A'];
# AscOption='toc'
# Limition='100';
def readConfig():
    # f = open('usr/src/config/config.txt')
    # f = open('usr/src/app/config.txt')
    f = open('config.txt')
    for line in f:
        configArr = line.split(' ')
        break
    return configArr

def SQLselect(selection,ismissing,tgtwidthSelect,tgtlengthSelect,tgtthicknessSelect,tocSelect,UpidSelect,Platetypes,AscOption,Limition):
    index=0;
    Process=['status_stats','status_cooling','status_furnace','status_rolling','status_fqc']
    miss=['0','0','0','0','0','0']
    missselect='';
    for i in ismissing:
        if(ismissing[i]):
            # selection.append(Process[index])
            missselect+= ' and '+i+'= '+miss[index];
        index+=1;
#         print(missselect)

    if(len(selection)==0): #select
        select='*'
        # print('fjdksjfskdjl')
    else:
        select = ','.join(selection)
#             print(select)
    if(len(tgtwidthSelect)==0):   #tgtwidth
        tgtwidth='';
    else: 
        tgtwidth=' and tgtwidth between '+tgtwidthSelect[0]+' and '+tgtwidthSelect[1]+' '
    if(len(tgtlengthSelect)==0):   #tgtlength
        tgtlength='';
    else: 
        tgtlength=' and tgtlength between '+tgtlengthSelect[0]+' and '+tgtlengthSelect[1]+' '
    if(len(tgtthicknessSelect)==0):  #tgtthickness
        tgtthickness='';
    else: 
        tgtthickness=' and tgtthickness between '+tgtthicknessSelect[0]+' and '+tgtthicknessSelect[1]+' '
    if(len(tocSelect)==0):   #toc
        toc='';
    else: 
        toc=' and toc between '+repr(tocSelect[0])+' and '+repr(tocSelect[1])+' '

    if(len(UpidSelect)==0): #upid
        upid='';
    else: 
        upid=' and (';
        for i in UpidSelect:
            upid+="upid="+repr(i)+' or '
        upid=upid[:-3]
        upid+=')';      
    if(len(Platetypes)==0):  #platetype
        platetype='';
    else: 
        platetype='and (';
        for i in Platetypes:
            platetype+='platetype='+repr(i)+' or '
        platetype=platetype[:-3]
        platetype+=')';
    if(AscOption==''):  #ASC
        ASC='';
    else:
        ASC=' ORDER BY '+AscOption+' DESC';
    if(Limition==''):  #Limit
        Limit='';
    else:
        Limit=' LIMIT '+Limition;
    Query=[tgtwidth,tgtlength,tgtthickness,toc,upid,platetype,missselect]

    for i in range(len(Query)):
        if(len(Query[i])!=0):
            Query[i]=Query[i][4:]
            # print(Query[i])
            SQL="select "+select+" from dcenter.dump_data " +"where"
            for j in Query:
                SQL+=j;
            SQL+=ASC+Limit;
            # print(SQL)
            return SQL
    return "select "+select+" from dcenter.dump_data "+ASC+Limit;

def getData(selection,ismissing,tgtwidthSelect,tgtlengthSelect,tgtthicknessSelect,tocSelect,UpidSelect,Platetypes,AscOption,Limition): 

    #  for docker outside config

    configArr = readConfig() 
    conn = psycopg2.connect(database=configArr[0], user=configArr[1], password=configArr[2], host=configArr[3],port=configArr[4])

    SQL=SQLselect(selection,ismissing,tgtwidthSelect,tgtlengthSelect,tgtthicknessSelect,tocSelect,UpidSelect,Platetypes,AscOption,Limition)

    # conn = psycopg2.connect(database='BSData20190713', user='postgres', password='616616', host='219.216.80.18',port='5432')
    # conn = psycopg2.connect(database='bg', user='postgres', password='woshimima', host='202.118.21.236',port='5432')

    cursor = conn.cursor()
    cursor.execute(SQL)
    rows = cursor.fetchall()
    
    conn.close()
    return rows
def SQLplateselect(selection,ismissing,tgtwidthSelect,tgtlengthSelect,tgtthicknessSelect,tocSelect,UpidSelect,Platetypes,AscOption,Limition):
    index=0;
    Process=['d.all_processes_statistics','d.v1','d.v2','d.v3','d.fqc_label']
    miss=['0','0','0','0','0','0']
    missselect='';
    for i in ismissing:
        if(ismissing[i]):
            # selection.append(Process[index])
            missselect+= ' and '+i+'= '+miss[index];
        index+=1;
#         print(missselect)

    if(len(selection)==0): #select
        select='*'
        # print('fjdksjfskdjl')
    else:
        select = ','.join(selection)
#             print(select)
    if(len(tgtwidthSelect)==0):   #tgtwidth
        tgtwidth='';
    else: 
        tgtwidth=' and d.tgtwidth between '+tgtwidthSelect[0]+' and '+tgtwidthSelect[1]+' '
    if(len(tgtlengthSelect)==0):   #tgtlength
        tgtlength='';
    else: 
        tgtlength=' and d.tgtlength between '+tgtlengthSelect[0]+' and '+tgtlengthSelect[1]+' '
    if(len(tgtthicknessSelect)==0):  #tgtthickness
        tgtthickness='';
    else: 
        tgtthickness=' and d.tgtthickness between '+tgtthicknessSelect[0]+' and '+tgtthicknessSelect[1]+' '
    if(len(tocSelect)==0):   #toc
        toc='';
    else: 
        toc=' and d.toc between '+repr(tocSelect[0])+' and '+repr(tocSelect[1])+' '

    if(len(UpidSelect)==0): #upid
        upid='';
    else: 
        upid=' and (';
        for i in UpidSelect:
            upid+="d.upid="+repr(i)+' or '
        upid=upid[:-3]
        upid+=')';      
    if(len(Platetypes)==0):  #platetype
        platetype='';
    else: 
        platetype='and (';
        for i in Platetypes:
            platetype+='m.productcategory='+repr(i)+' or '
        platetype=platetype[:-3]
        platetype+=')';
    if(AscOption==''):  #ASC
        ASC='';
    else:
        ASC=' ORDER BY d.'+AscOption+' DESC';
    if(Limition==''):  #Limit
        Limit='';
    else:
        Limit=' LIMIT '+Limition;
    Query=[tgtwidth,tgtlength,tgtthickness,toc,upid,platetype,missselect]
    index;
    for i in range(len(Query)):
        if(len(Query[i])!=0):
            Query[i]=Query[i][4:]
            # print(Query[i])
            SQL="select "+select+" from dcenter.dump_data d inner  join dcenter.l2_m_primary_data m ON d.upid=m.upid " +"where"
            for j in Query:
                SQL+=j;
            SQL+=ASC+Limit;
            # print(SQL)
            return getSQLData(SQL)
    return "select "+select+" from dcenter.dump_data "+ASC+Limit;
def getSQLData(SQLquery): 

    #  for docker outside config

    configArr = readConfig() 
    conn = psycopg2.connect(database=configArr[0], user=configArr[1], password=configArr[2], host=configArr[3],port=configArr[4])

    # conn = psycopg2.connect(database='BSData20190713', user='postgres', password='616616', host='219.216.80.18',port='5432')
    # conn = psycopg2.connect(database='bg', user='postgres', password='woshimima', host='202.118.21.236',port='5432')

    cursor = conn.cursor()
    cursor.execute(SQLquery)
    rows = cursor.fetchall()
    
    conn.close()
    return rows
def getFlagArr(str):
    arr = []
    midStr = re.findall('\[[0, 1].*[0,1]\]', str)
    m = re.search('([01]).*([01]).*([01]).*([01]).*([01])', midStr[0])
    for i in range(5):
        arr.append(int(m.group(i+1)))
    # print(arr)
    return arr
def getLabelData(SQLquery): 

    # conn = psycopg2.connect(database='BSData20190713', user='postgres', password='616616', host='219.216.80.18',port='5432')
    # conn = psycopg2.connect(database='bg', user='postgres', password='woshimima', host='202.118.21.236',port='5432')
    configArr = readConfig()
    conn = psycopg2.connect(database=configArr[0], user=configArr[1], password=configArr[2], host=configArr[3],port=configArr[4])

    cursor = conn.cursor()
    cursor.execute(SQLquery)
    rows = cursor.fetchall()

    # Extract the column names
    col_names = []
    for elt in cursor.description:
        col_names.append(elt[0])

    conn.close()
    return rows,col_names

def getLabel(data):
    label = ["头尾翘曲","中部厚度异常","中浪","左边浪","右边浪"]
    fault=getFlagArr(data['method1'])
    return dict(zip(label,fault))
ref = 5


def new_getData(selection, ismissing, tgtwidthSelect, tgtlengthSelect, tgtthicknessSelect, tocSelect, UpidSelect, Platetypes, AscOption, Limition):
    #  for docker outside config

    configArr = readConfig()
    conn = psycopg2.connect(database=configArr[0], user=configArr[1], password=configArr[2], host=configArr[3], port=configArr[4])

    SQL = sqlselect_bytime(selection, ismissing, tgtwidthSelect, tgtlengthSelect, tgtthicknessSelect, tocSelect, UpidSelect, Platetypes, AscOption, Limition)
    # print(SQL)
    # conn = psycopg2.connect(database='BSData20190713', user='postgres', password='616616', host='219.216.80.18',port='5432')
    # conn = psycopg2.connect(database='bg', user='postgres', password='woshimima', host='202.118.21.236',port='5432')

    cursor = conn.cursor()
    cursor.execute(SQL)
    rows = cursor.fetchall()
    col_names = []
    for elt in cursor.description:
        col_names.append(elt[0])
    conn.close()

    return rows, col_names

def new_SQLselect(selection,ismissing,tgtwidthSelect,tgtlengthSelect,tgtthicknessSelect,tocSelect,UpidSelect,Platetypes,AscOption,Limition):
    index=0
    miss=['0','0','0','0','0','0']
    missselect=''
    for i in ismissing:
        if(ismissing[i]):
            # selection.append(Process[index])
            missselect+= ' and '+i+'= '+miss[index]
        index+=1
#         print(missselect)

    if(len(selection)==0): #select
        select='*'
        # print('fjdksjfskdjl')
    else:
        select = ','.join(selection)
#             print(select)
    if(len(tgtwidthSelect)==0):   #tgtwidth
        tgtwidth=''
    else:
        tgtwidth=' and tgtwidth between '+tgtwidthSelect[0]+' and '+tgtwidthSelect[1]+' '
    if(len(tgtlengthSelect)==0):   #tgtlength
        tgtlength=''
    else:
        tgtlength=' and tgtlength between '+tgtlengthSelect[0]+' and '+tgtlengthSelect[1]+' '
    if(len(tgtthicknessSelect)==0):  #tgtthickness
        tgtthickness=''
    else:
        tgtthickness=' and tgtthickness between '+tgtthicknessSelect[0]+' and '+tgtthicknessSelect[1]+' '
    if(len(tocSelect)==0):   #toc
        toc=''
    else:
        toc=' and toc between '+repr(tocSelect[0])+' and '+repr(tocSelect[1])+' '

    if(len(UpidSelect)==0): #upid
        upid=''
    else:
        upid=' and ('
        for i in UpidSelect:
            upid+="upid="+repr(i)+' or '
        upid=upid[:-3]
        upid+=')'
    if(len(Platetypes)==0):  #platetype
        platetype=''
    else:
        platetype='and ('
        for i in Platetypes:
            platetype+='platetype='+repr(i)+' or '
        platetype=platetype[:-3]
        platetype+=')'
    if(AscOption==''):  #ASC
        ASC=''
    else:
        ASC=' ORDER BY '+AscOption+' DESC'
    if(Limition==''):  #Limit
        Limit=''
    else:
        Limit=' LIMIT '+Limition
    Query=[tgtwidth,tgtlength,tgtthickness,toc,upid,platetype,missselect]

    for i in range(len(Query)):
        if(len(Query[i])!=0):
            Query[i]=Query[i][4:]
            # print(Query[i])
            SQL="select "+select+" from app.deba_dump_data " +"where"
            for j in Query:
                SQL+=j
            SQL+=ASC+Limit
            # print(SQL)
            return SQL
    return "select "+select+" from app.deba_dump_data "+ASC+Limit


def sqlselect_bytime(selection, ismissing, tgtwidthSelect, tgtlengthSelect, tgtthicknessSelect, tocSelect, UpidSelect, Platetypes, AscOption, Limition):
    index=0
    miss=['0','0','0','0','0','0']
    missselect=''
    for i in ismissing:
        if(ismissing[i]):
            # selection.append(Process[index])
            missselect+= ' and '+i+'= '+miss[index]
        index+=1
#         print(missselect)

    if(len(selection)==0): #select
        select='*'
        # print('fjdksjfskdjl')
    else:
        select = ','.join(selection)
#             print(select)
    if(len(tgtwidthSelect)==0):   #tgtwidth
        tgtwidth=''
    else:
        tgtwidth=' and tgtwidth between '+tgtwidthSelect[0]+' and '+tgtwidthSelect[1]+' '
    if(len(tgtlengthSelect)==0):   #tgtlength
        tgtlength=''
    else:
        tgtlength=' and tgtlength between '+tgtlengthSelect[0]+' and '+tgtlengthSelect[1]+' '
    if(len(tgtthicknessSelect)==0):  #tgtthickness
        tgtthickness=''
    else:
        tgtthickness=' and tgtthickness between '+tgtthicknessSelect[0]+' and '+tgtthicknessSelect[1]+' '
    if(len(tocSelect)==0):   #toc
        toc=''
    else:
        toc=' and toc between '+repr(tocSelect[0])+' and '+repr(tocSelect[1])+' '

    if(len(UpidSelect)==0): #upid
        upid=''
    else:
        upid=' and ('
        for i in UpidSelect:
            upid+="upid="+repr(i)+' or '
        upid=upid[:-3]
        upid+=')'
    if(len(Platetypes)==0):  #platetype
        platetype=''
    else:
        platetype='and ('
        for i in Platetypes:
            platetype+='platetype='+repr(i)+' or '
        platetype=platetype[:-3]
        platetype+=')'
    if(AscOption==''):  #ASC
        ASC=''
    else:
        ASC=' ORDER BY '+AscOption+' DESC'
    if(Limition==''):  #Limit
        Limit=''
    else:
        Limit=' LIMIT '+Limition
    Query=[tgtwidth,tgtlength,tgtthickness,toc,upid,platetype,missselect]

    for i in range(len(Query)):
        if(len(Query[i])!=0):
            Query[i]=Query[i][4:]
            # print(Query[i])
            SQL="select "+select+" from app.deba_dump_data " +"where"
            for j in Query:
                SQL+=j
            SQL+=ASC+Limit
            # print(SQL)
            return SQL + " order by toc"
    return "select "+select+" from app.deba_dump_data "+ASC+Limit + " order by toc"


def new_getData(selection, ismissing, tgtwidthSelect, tgtlengthSelect, tgtthicknessSelect, tocSelect, UpidSelect, Platetypes, AscOption, Limition):
    #  for docker outside config

    configArr = readConfig()
    conn = psycopg2.connect(database=configArr[0], user=configArr[1], password=configArr[2], host=configArr[3], port=configArr[4])

    SQL = sqlselect_bytime(selection, ismissing, tgtwidthSelect, tgtlengthSelect, tgtthicknessSelect, tocSelect, UpidSelect, Platetypes, AscOption, Limition)
    # print("new_getData")
    # print(SQL)

    # conn = psycopg2.connect(database='BSData20190713', user='postgres', password='616616', host='219.216.80.18',port='5432')
    # conn = psycopg2.connect(database='bg', user='postgres', password='woshimima', host='202.118.21.236',port='5432')

    cursor = conn.cursor()
    cursor.execute(SQL)
    rows = cursor.fetchall()

    col_names = []
    for elt in cursor.description:
        col_names.append(elt[0])

    conn.close()
    return rows,col_names


def getData_bytime(selection, ismissing, tgtwidthSelect, tgtlengthSelect, tgtthicknessSelect, tocSelect, UpidSelect, Platetypes, AscOption, Limition):
    #  for docker outside config

    configArr = readConfig()
    conn = psycopg2.connect(database=configArr[0], user=configArr[1], password=configArr[2], host=configArr[3], port=configArr[4])

    SQL = sqlselect_bytime(selection, ismissing, tgtwidthSelect, tgtlengthSelect, tgtthicknessSelect, tocSelect, UpidSelect, Platetypes, AscOption, Limition)
    # print("new_getData")
    # print(SQL)

    # conn = psycopg2.connect(database='BSData20190713', user='postgres', password='616616', host='219.216.80.18',port='5432')
    # conn = psycopg2.connect(database='bg', user='postgres', password='woshimima', host='202.118.21.236',port='5432')

    cursor = conn.cursor()
    cursor.execute(SQL)
    rows = cursor.fetchall()

    col_names = []
    for elt in cursor.description:
        col_names.append(elt[0])

    conn.close()
    return rows,col_names
