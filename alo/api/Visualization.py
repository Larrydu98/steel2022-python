'''
Visualization
'''
from flask_restful import Resource, reqparse
from flask import json
from . import api
from ..controller.baogangPlot.createDiagResu import createDiagResu
import pika, traceback
from ..utils import getData,SQLplateselect,getFlagArr
import pandas as pd
import numpy as np
import json
from .singelSteel import lefttable,filterSQL,getLabelData
from scipy.interpolate import interp1d
import os
# path = os.getcwd()
parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class Visualization(Resource):
    '''
    Visualization
    '''
    def post(self,upid,process,deviation):
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
        SQL, status_cooling = filterSQL(parser)
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
            steel = {"min": list(np.percentile(indexData, middeviation, axis=0)),
                 "max": list(np.percentile(indexData, 100 - middeviation, axis=0)),
                 "emin": list(np.percentile(indexData, exdeviation, axis=0)),
                 "emax": list(np.percentile(indexData, 100 - exdeviation, axis=0)),
                 "mean": list(np.percentile(indexData, 50, axis=0)),
                 'sample': list(sampleData),
                "range": [indexData.min(), indexData.max()]}
            return steel
        UpidSelect=[]
        UpidSelect.append(upid)
        selection=[]
        selection1=[]
        if (process=='cool'):
          selection.append('v1')
          selection.append('v2')
          selection1.append('dd.v1')
          selection1.append('dd.v2')
        if (process=='heat'):
          selection.append('v2')
          selection1.append('dd.v2')
        if (process=='roll'):
          selection.append('v3')
          selection1.append('dd.v3')
        select = ','.join(selection1)
        ismissing = ['dd.all_processes_statistics_ismissing','dd.cool_ismissing','dd.fu_temperature_ismissing','dd.m_ismissing','dd.fqc_ismissing']
        if(SQL!=''):
            for i in ismissing:
                SQL+= ' and '+i+'= '+'0'
        if (SQL==''):
            SQL="where "+SQL
            for i in ismissing:
                SQL+= ' '+i+'= '+'0'+' and '
            SQL=SQL[:-4]
        SQL='select ' +select +lefttable + SQL
        data,col_names=getLabelData(SQL)
        print(SQL)
        # if(len(data) <50 ):
        #   SQL='select ' +select +lefttable + filterSteelSpec(parser)
        #   data,col_names=getLabelData(SQL)
        if(len(data) > 1000):
          data = data[0:1000]
        # data = SQLplateselect(selection1, ismissing, tgtwidthSelect, tgtlengthSelect, tgtthicknessSelect, [], [], Platetypes, 'toc', '1000')
        sampledata = getData(selection, {}, [], [], [], [], UpidSelect, [], '', '')
        jsondata={}
        len1=len(data)
        # print(len1)
        if (process=='cool'):
          name=['p1',"p2L",'p4','p5','p6']
          nameindex=[]
          coolright=1
          coolleft=1
          for m in range(len(name)):  ##冷却
            if(len(sampledata[0][0][name[m]]) != 0):
              nameindex.append(len(json.loads(sampledata[0][0][name[m]])['data']))
              coolKate=[]
              sampletemp=json.loads(sampledata[0][0][name[m]])['data']
              for i in range(len1):
                cooltemp=json.loads(data[i][0][name[m]])['data']
                if(len(cooltemp)>nameindex[m]+coolright):continue
                if(len(cooltemp)<nameindex[m]-coolleft):continue
                while(len(cooltemp)!=nameindex[m]):  #冷却插值
                    cooltemp=scipyutils(nameindex[m],cooltemp)
                coolKate.append(cooltemp)
              if (len(coolKate) == 0): continue
              coolKate=np.array(coolKate)
              jsondata[name[m]]=percentile(coolKate, sampletemp, deviation, limit)
          #二维温度场
          name1="Scanner"                
          coolKate=[]
          sampletemp=[]
          if(len(sampledata[0][1]['Scanner']) != 0):
            nameindex1=len(json.loads(sampledata[0][1]['Scanner'])['data'])
            sampletemp=np.array(json.loads(sampledata[0][1]['Scanner'])['data']).mean(axis=1)
            for i in range(len1):
              cooltemp=np.array(json.loads(data[i][1]['Scanner'])['data']).mean(axis=1)
              cooltemp=list(cooltemp)
              if(len(cooltemp)>nameindex1+coolright):continue
              if(len(cooltemp)<nameindex1-coolleft):continue
              while(len(cooltemp)!=nameindex1):  #冷却插值
                cooltemp=scipyutils(nameindex1,cooltemp)
              coolKate.append(cooltemp)
            # if (len(coolKate) == 0): continue
            coolKate=np.array(coolKate)
            jsondata[name1]=percentile(coolKate, sampletemp, deviation, limit)

        if (process=='heat'):
          name=['temp_seg_ul_1','temp_seg_ul_2','temp_seg_dl_s','temp_seg_ur_1','temp_seg_ur_2','temp_seg_dr_s']
          nameindex=[]         
          heatright=1
          heatleft=1
          for m in range(len(name)):#确定sampledata各项数据的指标
            nameindex.append(len(json.loads(sampledata[0][0]['Fufladc'])['data'][m]))
            heatKate=[]
            sampletemp=json.loads(sampledata[0][0]['Fufladc'])['data'][m]
            for i in range(len1): 
              if not data[i][0] is None:
                if(len(json.loads(data[i][0]['Fufladc'])['data']) != 0):
                  heattemp=json.loads(data[i][0]['Fufladc'])['data'][m]
                  if(len(heattemp)>=nameindex[m]+heatright):continue
                  if(len(heattemp)<=nameindex[m]-heatleft):continue
                  while(len(heattemp)!=nameindex[m]):  #插值
                    heattemp=scipyutils(nameindex[m],heattemp)
                  heatKate.append(heattemp)
            heatKate=np.array(heatKate)
            if(len(heatKate) == 0):continue
            # print(len(heatKate) == 0)
            jsondata[name[m]]=percentile(heatKate, sampletemp, deviation, limit)
        
        if(process=='roll'):
          name=["bendingforce","bendingforcebot","bendingforcetop","rollforce","rollforceds","rollforceos","screwdown","shiftpos","speed","torque","torquebot","torquetop"]
          name1=["contactlength","entrytemperature","exitflatness","exitprofile","exittemperature","exitthickness","exitwidth","forcecorrection"]      
          rollright=1
          rollleft=1
          nameindex=[]
          nameindex1=[]
          def rollfor(name,nameindex,serchkey):
            for m in range(len(name)):#确定sampledata各项数据的指标
              nameindex.append(len(json.loads(sampledata[0][0][serchkey])['data'][m]))
              rollKate=[]
              sampletemp=np.array(json.loads(sampledata[0][0][serchkey])['data'][m]).mean(axis=1)
              for i in range(len1): #n*k*8->n*k(sample)
                if not data[i][0] is None:
                  if(len(np.array(json.loads(data[i][0][serchkey])['data'][m]) != 0)):
                    rolltemp=list(np.array(json.loads(data[i][0][serchkey])['data'][m]).mean(axis=1)) #k*8->k
                    if(len(rolltemp)>nameindex[m]+rollright):
                      continue
                    if(len(rolltemp)<nameindex[m]-rollleft):
                      continue
                    # while(len(rolltemp)<nameindex[m]):
                    #   rolltemp.append(rolltemp[-1])
                    # while(len(rolltemp)>nameindex[m]):
                    #   rolltemp=rolltemp[:nameindex[m]]

                    rollKate.append(rolltemp)
              rollKate = np.array(rollKate)
              if(len(rollKate) == 0):continue;
              # print(rollKate.shape)
              # print(sampletemp.shape)
              # new_rollKate = np.array([])
              rollData = []
              # i=17
              steel = {"min": [],
                  "max": [],
                   "emin": [],
                    "emax": [],
                    "mean": [],
                    'sample': [],
                    "range": []}
              # print(name[m])
              for i in range(len(sampletemp)):
                  filter_rdata = []
                  if sampletemp[i] > 0:
                      # j=49
                      # for j in range(len(rollKate)):
                      # print('------------大于0--------------')
                      filter_rdatas = rollKate[rollKate.T[i].T > 0]
                      filter_rdata = filter_rdatas.T[i].T
                      rollData.append(filter_rdata)
                          # print(filter_rdata.shape)
                          # print(filter_rdata[0])
                  if sampletemp[i] < 0:


                      filter_rdatas = rollKate[rollKate.T[i].T < 0]
                      filter_rdata = filter_rdatas.T[i].T
                      rollData.append(filter_rdata)
                      if (name[m] == "torquebot"):
                          pass
                          # print('------------小于0--------------')
                          # print(filter_rdata)
                  if(len(filter_rdata) != 0):
                      steel["min"].append(np.percentile(filter_rdata, deviation, axis=0))
                      steel["max"].append(np.percentile(filter_rdata, 100 - deviation, axis=0))
                      steel["emin"].append(np.percentile(filter_rdata, limit, axis=0))
                      steel["emax"].append(np.percentile(filter_rdata, 100 - deviation, axis=0))
                      steel["mean"].append(np.percentile(filter_rdata, 50, axis=0))
                      steel["sample"].append(rolltemp[i])
              allRollData = []
              allRollData.append(steel["min"])
              allRollData.append(steel["sample"])
              allRollData.append(steel["max"])
              allRollData.append(steel["emin"])
              allRollData.append(steel["emax"])
              allRollData.append(steel["mean"])
              steel["range"]= [np.array(allRollData).min(), np.array(allRollData).max()]
              jsondata[name[m]]=steel
                  # percentile(np.array(rollData), sampletemp, deviation, limit)
              # def percentile(indexData, sampleData, middeviation, exdeviation):
              #     steel = {"min": list(np.percentile(indexData, middeviation, axis=0)),
              #              "max": list(np.percentile(indexData, 100 - middeviation, axis=0)),
              #              "emin": list(np.percentile(indexData, exdeviation, axis=0)),
              #              "emax": list(np.percentile(indexData, 100 - exdeviation, axis=0)),
              #              "mean": list(np.percentile(indexData, 50, axis=0)),
              #              'sample': list(sampleData),
              #              "range": [indexData.min(), indexData.max()]}
              #     return steel
          rollfor(name,nameindex,'meas')
          rollfor(name1,nameindex1,'post')

        for i in jsondata:
            singlejson = jsondata[i]
            for j in range(len(singlejson["sample"])):
                if(singlejson['min'][j] > singlejson['max'][j]):
                    singlejson['min'][j], singlejson['max'][j] = singlejson['max'][j], singlejson['min'][j]
                if (singlejson['emin'][j] > singlejson['emax'][j]):
                    singlejson['emin'][j], singlejson['emax'][j] = singlejson['emax'][j], singlejson['emin'][j]

        return jsondata,200, {'Access-Control-Allow-Origin': '*'}
api.add_resource(Visualization, '/v1.0/model/Visualization/<upid>/<process>/<deviation>')