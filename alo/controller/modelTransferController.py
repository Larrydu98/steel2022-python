'''
modelTransferController
'''
import pandas as pd
from  .getDataController import getDataController
from  .dataFilterController import dataFilterController
from  .dataFilterAfter import dataFilterAfter
# from controller.deepCNNSinTestController import deepCNNSinTestController
from  .deepCNNTestController import deepCNNTestController
from ..methods.dataPreMethods.FuFladcDataPre import FuFladcDataPre
from ..methods.dataPreMethods.MHpassMeasDataPre import MHpassMeasDataPre
from ..methods.dataPreMethods.MHpassPostDataPre import MHpassPostDataPre
from ..methods.dataPreMethods.MPgOutputDataPre import MPgOutputDataPre
from ..methods.dataPreMethods.SequenceDataPre import SequenceDataPre
from ..methods.dataPreMethods.Temperature2DPre import Temperature2DPre
from ..methods.dataPreMethods.TemperatureP12456Pre import TemperatureP12456Pre


class modelTransferController:
    '''
    modelTransferController
    '''
    def __init__(self, startTime, endTime):
        self.startTime = startTime
        self.endTime = endTime

    def run(self):
        '''
        run
        '''
        instance1 = getDataController(0, self.startTime, self.endTime)
        instance2 = getDataController(1, self.startTime, self.endTime)
        instance3 = getDataController(2, self.startTime,  self.endTime)
        instance4 = getDataController(3, self.startTime,  self.endTime)
        instance5 = getDataController(4, self.startTime,  self.endTime)
        instance6 = getDataController(5, self.startTime,  self.endTime)
        instance7 = getDataController(6, self.startTime,  self.endTime)
        # instance = pd.concat([data1,data2,data3,data4,data5,data6,data7])
        data1 = instance1.run()
        data2 = instance2.run()
        data3 = instance3.run()
        data4 = instance4.run()
        data5 = instance5.run()
        data6 = instance6.run()
        data7 = instance7.run()
        blockedData = {'Fu_M_Cc_Output': data1, 'Fu_fladcNew': data2,'M_Pg_Output_new': data3,
                        'M_Hpass_MeasNew': data4, 'hpass_post_raw_all': data5, 'raw_data_pdNew': data6, 'flag_df': data7}
        instanceFilter = dataFilterController(blockedData)
        
        mydata = instanceFilter.run()
        # print(mydata)
        instancePre1 = SequenceDataPre()
        dataPre1 = instancePre1.general_call(mydata)
        instancePre2 = FuFladcDataPre()
        dataPre2 = instancePre2.general_call(mydata)
        instancePre3 = MPgOutputDataPre()
        dataPre3 = instancePre3.general_call(mydata)
        instancePre4 = MHpassMeasDataPre()
        dataPre4 = instancePre4.general_call(mydata)
        instancePre5 = MHpassPostDataPre()
        dataPre5 = instancePre5.general_call(mydata)
        instancePre6 = Temperature2DPre()
        dataPre6 = instancePre6.general_call(mydata)
        instancePre7 = TemperatureP12456Pre()
        dataPre7 = instancePre7.general_call(mydata)
        afterBlockedData = {'Fu_M_Cc_Output':dataPre1, 'Fu_fladcNew': dataPre2, 'M_Pg_Output_new': dataPre3, 'M_Hpass_MeasNew': dataPre4,
        'hpass_post_raw_all': dataPre5, 'Temperature2DNew':dataPre6, 'TemperatureP12456New': dataPre7, 'flag_df': data7}
        afterInstanceFilter = dataFilterAfter(afterBlockedData)
        afterMyData = afterInstanceFilter.run()
        modelInstance = deepCNNTestController()
        modelData = modelInstance.run(afterMyData)
        return modelData
# instance = modelTransferController('2018-09-01 00:00:00', '2018-09-02 00:00:00')
# instance.run()