import psycopg2
import json
import copy
from ..utils import readConfig, allGetSQLData
from flask_restful import Resource, reqparse


class GetDiagnosisDataDB:

    # 获取训练集dataframe
    @staticmethod
    def getDaignosisTrainData(args, palte_type, plate_limit):
        SQLquery = '''
                select dd.upid,
                dd.platetype,
                dd.tgtwidth,
                dd.tgtlength,
                dd.tgtthickness,
                dd.stats,
                dd.fqc_label,
                dd.toc ,
                dd.status_fqc,
                dd.status_stats,
                lmpd.tgtdischargetemp,
                lmpd.tgttmplatetemp
               from  app.deba_dump_data dd
               right join dcenter.l2_m_primary_data lmpd on dd.upid = lmpd.upid 
                ''' + '''
                where {platetype}
                and {tgtthickness}
                and {tgtwidth}
                and {tgtlength}
                and {tgtdischargetemp}
                and {tgttmplatetemp}
                and {cooling}
                and dd.status_fqc = 0
                and dd.status_stats = 0
                ORDER BY dd.toc  DESC 
                Limit  {limit}'''.format(
            platetype = "dd.platetype = '" + str(palte_type) + "'",
            tgtthickness='1=1' if args["tgtthickness"] == [] else "dd.tgtthickness >= " + str(args["tgtthickness"][0]) + " and dd.tgtthickness <= " + str(args["tgtthickness"][1]),
            tgtwidth='1=1' if args["tgtwidth"] == [] else "dd.tgtwidth >= " + str(args["tgtwidth"][0]) + " and dd.tgtwidth <= " + str(args["tgtwidth"][1]),
            tgtlength='1=1' if args["tgtlength"] == [] else "dd.tgtlength >= " + str(args["tgtlength"][0]) + " and dd.tgtlength <= " + str(args["tgtlength"][1]),
            tgtdischargetemp='1=1' if args["tgtdischargetemp"] == [] else "lmpd.tgtdischargetemp >= " + str(args["tgtdischargetemp"][0]) + " and lmpd.tgtdischargetemp <= " + str(args["tgtdischargetemp"][1]),
            tgttmplatetemp='1=1' if args["tgttmplatetemp"] == [] else "lmpd.tgttmplatetemp >= " + str(args["tgttmplatetemp"][0]) + " and lmpd.tgttmplatetemp <= " + str(args["tgttmplatetemp"][1]),
            cooling='dd.status_cooling = ' + str(int(args['cooling'])),
            limit=int(plate_limit)
        )
        train_rows, train_col_names = allGetSQLData(SQLquery)
        return train_rows, train_col_names

    # and dd.status_stats = 0

    # 获取分类数据
    @staticmethod
    def getAllDiaData(start_time, end_time):
        SQLquery = '''
                select
                       dd.upid,
                       dd.platetype,
                       dd.tgtwidth,
                       dd.tgtlength,
                       dd.tgtthickness,
                       dd.toc,
                       dd.fqc_label as label,
                       (fqc_label->>'method1')::json->>'data' as flag_lable,
                       dd.status_fqc,
                       dd.stats,
                       dd.status_stats,
                       dd.status_cooling,
                       lmpd.tgtdischargetemp,
                       lmpd.tgttmplatetemp
                       from  app.deba_dump_data dd
                       right join dcenter.l2_m_primary_data lmpd on dd.upid = lmpd.upid 
               ''' + '''
               where {start_time} 
               and {end_time}
               order by dd.toc
               '''.format(start_time='1=1' if start_time == 'all' else "dd.toc >= to_timestamp('" + str(start_time) + "','yyyy-mm-dd hh24:mi:ss') ",
                          end_time='1=1' if end_time == 'all' else "dd.toc <= to_timestamp('" + str(end_time) + "','yyyy-mm-dd hh24:mi:ss') ")
        rows, col_names = allGetSQLData(SQLquery)
        return rows, col_names
