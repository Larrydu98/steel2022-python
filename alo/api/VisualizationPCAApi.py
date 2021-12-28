'''
VisualizationPCAApi
'''
from flask_restful import Resource, reqparse
from . import api
import pandas as pd
from ..controller.VisualizationPCAController import getVisualizationPCA
from .singelSteel import modeldata,mareymodeldata,thicklabel, cate_modeldata
from ..api import singelSteel

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

# # 根目录
# @app.route('/')


class VisualizationPCA(Resource):
    '''
    SixDpictureUpDownQuantile
    '''
    def post(self, startTime, endTime):
        """
        get
        ---
        tags:
          - 可视化马雷图部分PCA
        parameters:
            - in: path
              name: startTime
              required: true
              description: 开始钢板upid
              type: string
            - in: path
              name: endTime
              required: true
              description: 结束钢板upid
              type: string
        responses:
            200:
                description: 执行成功
        """
        data, status_cooling = modeldata(parser,
                                        ['dd.upid', 'lmpd.steelspec', 'dd.toc', 'dd.tgtwidth', 'dd.tgtlength', 'dd.tgtthickness * 1000 as tgtthickness',
                                         'dd.stats', 'dd.fqc_label', thicklabel, 'dd.status_cooling', 'dd.status_fqc',
                                         'lmpd.slabthickness * 1000 as slabthickness', 'lmpd.tgtdischargetemp', 'lmpd.tgttmplatetemp',
                                         'lcp.cooling_start_temp', 'lcp.cooling_stop_temp', 'lcp.cooling_rate1'],
                                        startTime,
                                        endTime)
        # marey_data, col_names = mareymodeldata(start_time, end_time, parser,
        #                                        ['dd.upid', 'lmpd.steelspec', 'dd.toc', 'dd.tgtwidth', 'dd.tgtlength', 'dd.tgtthickness', 'dd.stats', 'dd.fqc_label', thicklabel,
        #                                         'dd.status_fqc'],
        #                                        status_cooling)
        # data_df = pd.DataFrame(data=data, columns=col_names).dropna(axis=0, how='any').reset_index(drop=True)
        # marey_data_df = pd.DataFrame(data=marey_data, columns=col_names).dropna(axis=0, how='any').reset_index(drop=True)
        # data = data_df.append(marey_data_df).drop_duplicates(['upid']).reset_index(drop=True).values.tolist()

        if len(data) <= 1:
            return {}, 204, {'Access-Control-Allow-Origin': '*'}

        visualizationPCA = getVisualizationPCA()

        # data_names = []
        # if status_cooling == 0:
        #     data_names = singelSteel.data_names
        # elif status_cooling == 1:
        #     data_names = singelSteel.without_cooling_data_names
        json = visualizationPCA.run(data)

        if len(json) < 5:
            return json, 202, {'Access-Control-Allow-Origin': '*'}
        return json, 200, {'Access-Control-Allow-Origin': '*'}


class CateVisualizationPCA(Resource):
    def post(self, startTime, endTime):
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

        visualizationPCA = getVisualizationPCA()

        json = visualizationPCA.cate_run(data, data_1, data_2)

        if len(json) < 5:
            return json, 202, {'Access-Control-Allow-Origin': '*'}
        return json, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(VisualizationPCA, '/v1.0/model/VisualizationPCA/<startTime>/<endTime>/')
api.add_resource(CateVisualizationPCA, '/v1.0/model/CateVisualizationPCA/<startTime>/<endTime>/<limition>')
