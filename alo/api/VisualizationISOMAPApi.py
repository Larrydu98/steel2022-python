'''
VisualizationMDSApi
'''
from flask_restful import Resource, reqparse
from . import api
import pandas as pd
from ..controller.VisualizationISOMAPController import getVisualizationISOMAP
from .singelSteel import modeldata,mareymodeldata,thicklabel
from ..api import singelSteel

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

class VisualizationISOMAP(Resource):
    def post(self, startTime, endTime):
        data, status_cooling = modeldata(parser,
                                        ['dd.upid', 'lmpd.steelspec', 'dd.toc', 'dd.tgtwidth', 'dd.tgtlength', 'dd.tgtthickness * 1000 as tgtthickness',
                                         'dd.stats', 'dd.fqc_label', thicklabel, 'dd.status_cooling', 'dd.status_fqc',
                                         'lmpd.slabthickness * 1000 as slabthickness', 'lmpd.tgtdischargetemp', 'lmpd.tgttmplatetemp',
                                         'lcp.cooling_start_temp', 'lcp.cooling_stop_temp', 'lcp.cooling_rate1'],
                                        startTime,
                                        endTime)

        if len(data) <= 1:
            return {}, 204, {'Access-Control-Allow-Origin': '*'}

        visualizationISOMAP = getVisualizationISOMAP()

        # data_names = []
        # if status_cooling == 0:
        #     data_names = singelSteel.data_names
        # elif status_cooling == 1:
        #     data_names = singelSteel.without_cooling_data_names
        json = visualizationISOMAP.run(data)

        if len(json) < 5:
            return json, 202, {'Access-Control-Allow-Origin': '*'}
        return json, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(VisualizationISOMAP, '/v1.0/model/VisualizationISOMAP/<startTime>/<endTime>/')