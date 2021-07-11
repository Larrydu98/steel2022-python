'''
Visualization
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
import pandas as pd
import numpy as np
import json
from .singelSteel import filterSQL,getLabelData, new_filterSQL
from ..controller.RollingPassStatisticsController import RollingPassStatistics
from scipy.interpolate import interp1d
parser = reqparse.RequestParser(trim=True, bundle_errors=True)
lefttable = ''' from  dcenter.l2_m_primary_data lmpd
            left join dcenter.l2_fu_acc_t lfat on lmpd.slabid = lfat.slab_no
            left join dcenter.l2_m_plate lmp   on lmpd.slabid = lmp.slabid
            left join dcenter.l2_cc_pdi lcp    on lmpd.slabid = lcp.slab_no
            left join app.deba_dump_data dd   on dd.upid = lmp.upid '''

class newVisualization(Resource):
    '''
    Visualization
    '''
    def post(self, upid, process, deviation, limitation):
        """
        post
        ---
        tags:
            - 诊断视图数据
        parameters:
            - in: body
              name: body
              schema:
              properties:
                  upid:
                #   type: string
                #   default: 1
                #   description: 数据预处理
                #   task_id:
                #   type: string
                #   default: 1
                #   description: 任务id
              required: true
        responses:
          200:
            description: 执行成功
        """
        SQL, status_cooling, fqcflag = new_filterSQL(parser)
        deviation= 100 * float(deviation)
        limit = 5
        def eyearray(tempnum):
            return np.linspace(0,10*np.pi,num=tempnum)
        def scipyutils(num,x_diff):
            x=eyearray(num)
            y=x_diff
            x_diff=eyearray(len(x_diff))
            f1=interp1d(x_diff,y,kind='linear')#线性插值
            # f2=interp1d(x_diff,y,kind='cubic')#三次样条插值
            return f1(x)
        def percentile(indexData, sampleData, middeviation, exdeviation):
            # aaa = np.array(sampleData).reshape(1,len(sampledata))
            # range_array = np.vstack((indexData, aaa))
            # print(len(indexData))
            # print(len(indexData[0]))
            # print(len(sampleData))
            indexData = np.append(indexData,np.array(sampleData).reshape(1,len(sampleData)), axis=0)
            steel = {"min": list(np.percentile(indexData, middeviation, axis=0)),
                 "max": list(np.percentile(indexData, 100 - middeviation, axis=0)),
                 "emin": list(np.percentile(indexData, exdeviation, axis=0)),
                 "emax": list(np.percentile(indexData, 100 - exdeviation, axis=0)),
                 "mean": list(np.percentile(indexData, 50, axis=0)),
                 'sample': list(sampleData),
                "range": [indexData.min(), indexData.max()]}
            return steel
        # UpidSelect=[]
        # UpidSelect.append(upid)
        selection=[]
        ismissing = []
        if (process=='cool'):
            selection = ["dd.cooling",'dd.fqc_label','dd.status_cooling']
            if fqcflag == 0:
                ismissing = "dd.status_cooling = 0 and dd.status_fqc = 0"
            elif fqcflag == 1:
                ismissing = "dd.status_cooling = 0"
        if (process=='heat'):
            selection = ["dd.furnace",'dd.fqc_label','dd.status_furnace']
            if fqcflag == 0:
                ismissing = "dd.status_furnace = 0 and dd.status_fqc = 0"
            elif fqcflag == 1:
                ismissing = "dd.status_furnace = 0"
        if (process=='roll'):
            selection = ["dd.rolling",'dd.fqc_label','dd.status_rolling']
            if fqcflag == 0:
                ismissing = "dd.status_rolling = 0 and dd.status_fqc = 0"
            elif fqcflag == 1:
                ismissing = "dd.status_rolling = 0"
        select = ','.join(selection)
        if(SQL!=''):
            SQL+= ' and '+ismissing
        if (SQL==''):
            SQL="where "+ismissing
        SQL='select ' + select + lefttable + SQL + ' ORDER BY dd.toc  DESC Limit ' + str(limitation)
        # print(SQL)
        data, col_names = getLabelData(SQL)
        # print(len(data))
        goodData = []

        if fqcflag == 0:
            for item in data:
                flags = item[1]['method1']['data']
                if np.array(flags).sum() == 5:
                    goodData.append(item)
        elif fqcflag == 1:
            goodData = data

        # print(len(goodData))
        data = pd.DataFrame(data=goodData, columns=col_names).dropna(axis=0, how='any').reset_index(drop=True)
        # print(data)

        # data = SQLplateselect(selection1, ismissing, tgtwidthSelect, tgtlengthSelect, tgtthicknessSelect, [], [], Platetypes, 'toc', '1000')
        sampledata,col_names = getLabelData('select ' + select +lefttable + "where dd.upid = " + repr(upid))
        sampledata = pd.DataFrame(data=sampledata, columns=col_names).dropna(axis=0, how='any').reset_index(drop=True)

        jsondata={}
        len1=len(data)

        if len1 == 0:
            return {}, 204, {'Access-Control-Allow-Origin': '*'}

        if (process=='cool'):

            if sampledata.status_cooling.values[0] == 1:
                return {}, 204, {'Access-Control-Allow-Origin': '*'}

            name=['p1',"p2",'p3','p4','p6']
            nameindex=[]
            coolright=1
            coolleft=1
            for m in range(len(name)):  ##冷却
                nameindex.append(len(sampledata.iloc[0, :].values[0]['temp'][name[m]]['data']))
                coolKate=[]
                sampletemp=sampledata.iloc[0, :].values[0]['temp'][name[m]]['data']
                sampleposition = sampledata.iloc[0, :].values[0]['temp'][name[m]]['position']
                for i in range(len1):
                    cooltemp=data.iloc[i, :].values[0]['temp'][name[m]]['data']
                    if(len(cooltemp)>nameindex[m]+coolright):continue
                    if(len(cooltemp)<nameindex[m]-coolleft):continue
                    while(len(cooltemp)!=nameindex[m]):  #冷却插值
                        cooltemp=scipyutils(nameindex[m],cooltemp)
                    coolKate.append(cooltemp)
                if (len(coolKate) == 0): continue
                coolKate=np.array(coolKate)
                jsondata[name[m]]=percentile(coolKate, sampletemp, deviation, limit)
                sampleposition_p = np.abs(np.array(sampleposition)) / sampleposition[-1] * 100
                jsondata[name[m]]['position'] = sampleposition_p.tolist()
            # 二维温度场
            name1="Scanner"
            coolKate=[]
            sampletemp=[]
            nameindex1=len(sampledata.iloc[0,:].values[0]['scanner']['data'][0])
            sampletemp = np.array(sampledata.iloc[0,:].values[0]['scanner']['data']).T.mean(axis=1)
            sampleposition = np.array(sampledata.iloc[0, :].values[0]['scanner']['position'])#.mean(axis=1)
            for i in range(len1):
                # print(i)
                cooltemp=np.array(data.iloc[i,:].values[0]['scanner']['data']).T.mean(axis=1)
                cooltemp=list(cooltemp)
                # if(len(cooltemp)>nameindex1+coolright):continue
                # if(len(cooltemp)<nameindex1-coolleft):continue
                while(len(cooltemp)!=nameindex1):  #冷却插值
                    cooltemp=scipyutils(nameindex1,cooltemp)
                coolKate.append(cooltemp)
            # if (len(coolKate) == 0): continue
            coolKate=np.array(coolKate)
            jsondata[name1]=percentile(coolKate, sampletemp, deviation, limit)
            jsondata[name1]['position'] = (np.abs(np.array(sampleposition)) / sampleposition[-1] * 100).tolist()


        if (process=='heat'):

            if sampledata.status_furnace.values[0] == 1:
                return {}, 204, {'Access-Control-Allow-Origin': '*'}

            name=['seg_u','seg_d','plate', 'time']
            nameindex=[]
            heatright=1
            heatleft=1
            # json.loads(sampledata[0][0]['Fufladc'])['data'][m]
            # print(sampledata.iloc[0, :].values[0]['temp_seg_ul_1'])
            for m in range(len(name)):#确定sampledata各项数据的指标
                nameindex.append(len(sampledata.iloc[0, :].values[0][name[m]]))
                heatKate=[]
                sampletemp=sampledata.iloc[0, :].values[0][name[m]]
                for i in range(len1):
                    # if not data[i][0] is None:
                    #   if(len(json.loads(data[i][0]['Fufladc'])['data']) != 0):
                    heattemp=data.iloc[i, :].values[0][name[m]]
                    # if(len(heattemp)>=nameindex[m]+heatright):continue
                    # if(len(heattemp)<=nameindex[m]-heatleft):continue
                    while(len(heattemp)!=nameindex[m]):  #插值
                        heattemp=scipyutils(nameindex[m],heattemp)
                    heatKate.append(heattemp)
                heatKate=np.array(heatKate)
                # if(len(heatKate) == 0):continue
                # print(len(heatKate) == 0)
                jsondata[name[m]] = percentile(heatKate, sampletemp, deviation, limit)
                samplepostion_heat = sampledata.iloc[0, :].values[0]['position']
                jsondata[name[m]]['position'] = samplepostion_heat


        if(process=='roll'):

            if sampledata.status_rolling.values[0] == 1:
                return {}, 204, {'Access-Control-Allow-Origin': '*'}

            name=["bendingforce", "bendingforcebot", "bendingforcetop", "rollforce", "rollforceds", "rollforceos",
                  "screwdown", "shiftpos", "speed", "torque", "torquebot", "torquetop"]
            name1=["contactlength", "entrytemperature", "exitflatness", "exitprofile", "exittemperature",
                   "exitthickness", "exitwidth", "forcecorrection"]
            rollright=1
            rollleft=1
            nameindex=[]
            nameindex1=[]
            def rollfor(name,nameindex,serchkey):
                for m in range(len(name)):#确定sampledata各项数据的指标
                    nameindex.append(len(sampledata.iloc[0,:].values[0][serchkey][name[m]]))
                    rollKate=[]
                    sampletemp=np.array(sampledata.iloc[0,:].values[0][serchkey][name[m]]).mean(axis=1)
                    for i in range(len1): #n*k*8->n*k(sample)
                        # print(np.array(data.iloc[i,:].values[0][serchkey][name[m]]).shape)
                        # print(data.iloc[i, :].values[-1])
                        # if()
                        rolltemp=list(np.array(data.iloc[i,:].values[0][serchkey][name[m]]).mean(axis=1)) #k*8->k
                        # print(rolltemp)
                        if(len(rolltemp)!=nameindex[m]):continue
                        # if(len(rolltemp)=<nameindex[m]-rollleft):continue
                        rollKate.append(rolltemp)
                    rollKate = np.array(rollKate)
                    if(len(rollKate) == 0):continue
                    rollData = []
                    # i=17
                    steel = {"min": [], "max": [], "emin": [], "emax": [], "mean": [], 'sample': [], "range": []}
                    if(name[m] == "shiftpos"):
                        for i in range(len(sampletemp)):
                            filter_rdata = []
                            if sampletemp[i] > 0:
                                # j=49
                                # for j in range(len(rollKate)):
                                # print('------------大于0--------------')
                                filter_rdatas = rollKate[rollKate.T[i].T > 0]
                                filter_rdata = filter_rdatas.T[i].T
                                # rollData.append(filter_rdata)
                            if sampletemp[i] < 0:
                                filter_rdatas = rollKate[rollKate.T[i].T < 0]
                                filter_rdata = filter_rdatas.T[i].T
                                # rollData.append(filter_rdata)
                                # if (name[m] == "torquebot"):
                                #     pass
                                    # print('------------小于0--------------')
                                    # print(filter_rdata)
                            if(len(filter_rdata) != 0):
                                steel["min"].append(np.percentile(filter_rdata, deviation, axis=0))
                                steel["max"].append(np.percentile(filter_rdata, 100 - deviation, axis=0))
                                steel["emin"].append(np.percentile(filter_rdata, limit, axis=0))
                                steel["emax"].append(np.percentile(filter_rdata, 100 - deviation, axis=0))
                                steel["mean"].append(np.percentile(filter_rdata, 50, axis=0))
                                steel["sample"].append(sampletemp[i])
                    if (name[m] != "shiftpos"):
                        for i in range(len(sampletemp)):
                            filter_rdata = []
                            if sampletemp[i] > 0:
                                # j=49
                                # for j in range(len(rollKate)):
                                # print('------------大于0--------------')
                                filter_rdatas = rollKate[rollKate.T[i].T > 0]
                                filter_rdata = filter_rdatas.T[i].T
                                # rollData.append(filter_rdata)
                            if sampletemp[i] < 0:
                                filter_rdatas = rollKate[rollKate.T[i].T < 0]
                                filter_rdata = np.absolute(filter_rdatas.T[i].T)
                                # rollData.append(filter_rdata)
                                # if (name[m] == "torquebot"):
                                #     pass
                                    # print('------------小于0--------------')
                                    # print(filter_rdata)
                            if(len(filter_rdata) != 0):
                                steel["min"].append(np.percentile(filter_rdata, deviation, axis=0))
                                steel["max"].append(np.percentile(filter_rdata, 100 - deviation, axis=0))
                                steel["emin"].append(np.percentile(filter_rdata, limit, axis=0))
                                steel["emax"].append(np.percentile(filter_rdata, 100 - deviation, axis=0))
                                steel["mean"].append(np.percentile(filter_rdata, 50, axis=0))
                                steel["sample"].append(abs(sampletemp[i]))
                    allRollData = []
                    allRollData.append(steel["min"])
                    allRollData.append(steel["sample"])
                    allRollData.append(steel["max"])
                    allRollData.append(steel["emin"])
                    allRollData.append(steel["emax"])
                    allRollData.append(steel["mean"])
                    allRollData.append(list(steel["sample"]))
                    # indexData = np.append(indexData, np.array(sampleData).reshape(1, len(sampleData)), axis=0)

                    steel["range"]= [np.array(allRollData).min(), np.array(allRollData).max()]
                    jsondata[name[m]]=steel
            rollfor(name,nameindex,'meas')

        for i in jsondata:
            singlejson = jsondata[i]
            for j in range(len(singlejson["sample"])):
                sortArray = [singlejson['min'][j], singlejson['emin'][j], singlejson['mean'][j], singlejson['max'][j],singlejson['emax'][j]]
                sortArray.sort()
                singlejson['emin'][j] = sortArray[0]
                singlejson['min'][j] = sortArray[1]
                singlejson['mean'][j] = sortArray[2]
                singlejson['max'][j] = sortArray[3]
                singlejson['emax'][j] = sortArray[4]
                # singlejson
        #         if(singlejson['min'][j] >= singlejson['max'][j]) or ():
                #     singlejson['min'][j], singlejson['max'][j] = singlejson['max'][j], singlejson['min'][j]
                # if (singlejson['emin'][j] > singlejson['emax'][j]):
                #     singlejson['emin'][j], singlejson['emax'][j] = singlejson['emax'][j], singlejson['emin'][j]
        if len(jsondata) < 5:
            return jsondata, 202, {'Access-Control-Allow-Origin': '*'}
        return jsondata,200, {'Access-Control-Allow-Origin': '*'}


class RollingPassStatisticsApi(Resource):
    def post(self, limitation):
        selection = ["dd.rolling", 'dd.fqc_label', 'dd.status_fqc']
        ismissing = "dd.status_rolling = 0 "

        rollingPass_instance = RollingPassStatistics()
        data, columns = rollingPass_instance.getData(parser, selection, lefttable, ismissing, limitation)

        status_code, result = rollingPass_instance.getRollingPassStatistics(data)

        return result, status_code, {'Access-Control-Allow-Origin': '*'}


api.add_resource(newVisualization, '/v1.0/model/newVisualization/<upid>/<process>/<deviation>/<limitation>')
api.add_resource(RollingPassStatisticsApi, '/v1.0/model/RollingPassStatisticsApi/<limitation>')