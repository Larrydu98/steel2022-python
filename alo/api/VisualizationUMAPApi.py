'''
VisualizationUMAPApi
'''
from flask_restful import Resource, reqparse
from . import api
import pandas as pd
from ..controller.VisualizationUMAPController import getVisualizationUMAP
from .singelSteel import modeldata, mareymodeldata, thicklabel, cate_modeldata
from ..api import singelSteel

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

class VisualizationUMAP(Resource):
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

        visualizationUMAP = getVisualizationUMAP()

        # data_names = []
        # if status_cooling == 0:
        #     data_names = singelSteel.data_names
        # elif status_cooling == 1:
        #     data_names = singelSteel.without_cooling_data_names
        json = visualizationUMAP.run(data)

        if len(json) < 5:
            return json, 202, {'Access-Control-Allow-Origin': '*'}
        return json, 200, {'Access-Control-Allow-Origin': '*'}


class CateVisualizationUMAP(Resource):
    def post(self, startTime, endTime, limition):
        data, status_cooling = modeldata(parser,
                                        ['dd.upid', 'lmpd.steelspec', 'dd.toc', 'dd.tgtwidth', 'dd.tgtlength', 'dd.tgtthickness * 1000 as tgtthickness',
                                         'dd.stats', 'dd.fqc_label', thicklabel, 'dd.status_cooling', 'dd.status_fqc',
                                         'lmpd.slabthickness * 1000 as slabthickness', 'lmpd.tgtdischargetemp', 'lmpd.tgttmplatetemp',
                                         'lcp.cooling_start_temp', 'lcp.cooling_stop_temp', 'lcp.cooling_rate1'],
                                        startTime,
                                        endTime)

        limit_1 = ''' order by abs(extract(epoch from dd.toc - '{center_time}'::timestamp))
                                limit {limition}; '''.format(center_time=startTime, limition=limition)
        data_1, status_cooling = cate_modeldata(parser,
                                                ['dd.upid', 'lmpd.steelspec', 'dd.toc', 'dd.tgtwidth', 'dd.tgtlength', 'dd.tgtthickness * 1000 as tgtthickness',
                                                 'dd.stats', 'dd.fqc_label', thicklabel, 'dd.status_cooling', 'dd.status_fqc',
                                                 'lmpd.slabthickness * 1000 as slabthickness', 'lmpd.tgtdischargetemp', 'lmpd.tgttmplatetemp',
                                                 'lcp.cooling_start_temp', 'lcp.cooling_stop_temp', 'lcp.cooling_rate1'],
                                                limit_1)

        limit_2 = ''' order by abs(extract(epoch from dd.toc - '{center_time}'::timestamp))
                                limit {limition}; '''.format(center_time=endTime, limition=limition)
        data_2, status_cooling = cate_modeldata(parser,
                                                ['dd.upid', 'lmpd.steelspec', 'dd.toc', 'dd.tgtwidth', 'dd.tgtlength', 'dd.tgtthickness * 1000 as tgtthickness',
                                                 'dd.stats', 'dd.fqc_label', thicklabel, 'dd.status_cooling', 'dd.status_fqc',
                                                 'lmpd.slabthickness * 1000 as slabthickness', 'lmpd.tgtdischargetemp', 'lmpd.tgttmplatetemp',
                                                 'lcp.cooling_start_temp', 'lcp.cooling_stop_temp', 'lcp.cooling_rate1'],
                                                limit_2)

        if len(data) <= 1:
            return {}, 204, {'Access-Control-Allow-Origin': '*'}

        visualizationUMAP = getVisualizationUMAP()

        json = visualizationUMAP.cate_run(data, data_1, data_2)

        if len(json) < 5:
            return json, 202, {'Access-Control-Allow-Origin': '*'}
        return json, 200, {'Access-Control-Allow-Origin': '*'}

api.add_resource(VisualizationUMAP, '/v1.0/model/VisualizationUMAP/<startTime>/<endTime>/')
api.add_resource(CateVisualizationUMAP, '/v1.0/model/CateVisualizationUMAP/<startTime>/<endTime>/<limition>')
