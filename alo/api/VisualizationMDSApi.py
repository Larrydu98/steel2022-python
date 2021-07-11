'''
VisualizationMDSApi
'''
from flask_restful import Resource, reqparse
from . import api
import pandas as pd
from ..controller.VisualizationMDSController import getVisualizationMDS
from .singelSteel import modeldata,mareymodeldata,thicklabel
from ..api import singelSteel

parser = reqparse.RequestParser(trim=True, bundle_errors=True)

class VisualizationMDS(Resource):
    '''
    VisualizationMDS
    '''
    def post(self, startTime, endTime):
        """
        get
        ---
        tags:
			- 可视化马雷图部分MDS
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
        # marey_data, col_names = mareymodeldata(start_time, end_time, parser,['dd.upid', 'lmpd.steelspec','dd.toc', 'dd.tgtwidth','dd.tgtlength','dd.tgtthickness','dd.stats','dd.fqc_label',thicklabel,'dd.status_fqc'], status_cooling)
        # data_df = pd.DataFrame(data=data, columns=col_names).dropna(axis=0, how='any').reset_index(drop=True)
        # marey_data_df = pd.DataFrame(data=marey_data, columns=col_names).dropna(axis=0, how='any').reset_index(drop=True)
        # data = data_df.append(marey_data_df).drop_duplicates(['upid']).reset_index(drop=True).values.tolist()

        if len(data) <= 1:
            return {}, 204, {'Access-Control-Allow-Origin': '*'}

        visualizationMDS = getVisualizationMDS()

        # data_names = []
        # if status_cooling == 0:
        #     data_names = singelSteel.data_names
        # elif status_cooling == 1:
        #     data_names = singelSteel.without_cooling_data_names
        json = visualizationMDS.run(data)

        if len(json) < 5:
            return json, 202, {'Access-Control-Allow-Origin': '*'}
        return json, 200, {'Access-Control-Allow-Origin': '*'}


api.add_resource(VisualizationMDS, '/v1.0/model/VisualizationMDS/<startTime>/<endTime>/')