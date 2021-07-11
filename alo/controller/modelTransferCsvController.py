'''
modelTransferCsvController
'''
import pandas as pd
from  .getDataController import getDataController
from  .dataFilterController import dataFilterController
from  .dataFilterCsvAfter import dataFilterCsvAfter
from  .dataFilterCsvAfter import dataFilterCsvAfterAll
from  .dataFilterCsvAfter import dataFilterCsvAfterByTime
from  .deepCNNSinTestControllerNew import deepCNNSinTestController
# from controller.deepCNNSinTestController import deepCNNSinTestController
from  .deepCNNTestController import deepCNNTestController
from  .getDataController import getDataVisualizationMaretoUpidFlagController
from  .deepCNNSinTestControllerNew import deepCNNSinTestController
from  .deepCNNSinTestControllerNew import deepCNNOnePlatePredictController
from  .deepCNNSinTestControllerNew import deepCNNPlatePredictByTimeController


class modelTransferCsvController:
    '''
    modelTransferCsv
    '''
    def __init__(self, startTime, endTime):
        self.startTime = startTime
        self.endTime = endTime

    def run(self):
        '''
        run
        '''
        afterInstanceFilter = dataFilterCsvAfter()
        afterMyData = afterInstanceFilter.run()
        modelInstance = deepCNNTestController()
        modelData = modelInstance.run(afterMyData)
        return modelData
# instance = modelTransferController('2018-09-01 00:00:00', '2018-09-02 00:00:00')
# instance.run()


class modelTransferCsvforSomePlate:
    '''
    modelTransferCsvforSomePlate
    '''
    def __init__(self, startTime, endTime):
        self.startTime = startTime
        self.endTime = endTime

    def run(self):
        '''
        run
        '''
        afterInstanceFilter = dataFilterCsvAfterAll()
        afterMyData = afterInstanceFilter.run()
        VisualizationMaretoUpidFlag = getDataVisualizationMaretoUpidFlagController(self.startTime,self.endTime)
        VisualizationMaretoUpidFlagData = VisualizationMaretoUpidFlag.run()
        modelInstance = deepCNNSinTestController()
        modelData = modelInstance.run(afterMyData,VisualizationMaretoUpidFlagData)
        return modelData

class modelTransferCsvforOnePlate:
    '''
    modelTransferCsvforOnePlate
    '''
    def __init__(self, upid):
        self.upid = upid

    def run(self):
        '''
        run
        '''
        afterInstanceFilter = dataFilterCsvAfterAll()
        afterMyData = afterInstanceFilter.run()
        modelInstance = deepCNNOnePlatePredictController()
        modelData = modelInstance.run(afterMyData,self.upid)
        return modelData

class modelTransferCsvforPlateByTime:
    '''
    modelTransferCsvforPlateByTime
    '''
    def __init__(self, startTime,endTime):
        self.startTime = startTime
        self.endTime = endTime

    def run(self):
        '''
        run
        '''
        afterInstanceFilter = dataFilterCsvAfterByTime()
        afterMyData = afterInstanceFilter.run(self.startTime,self.endTime)
        modelInstance = deepCNNPlatePredictByTimeController()
        modelData = modelInstance.run(afterMyData)
        return modelData