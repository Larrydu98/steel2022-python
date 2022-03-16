import psycopg2
import json
from ..utils import readConfig, allGetSQLData
from flask_restful import Resource, reqparse

parser = reqparse.RequestParser(trim=True, bundle_errors=True)


class GetDiagnosisData:
    def __init__(self, args, plate_limit):

        self.args = args
        # self.args['upids'] =  self.args['upids'].replace('[', '(').replace(']', ')')
        self.plate_limit = plate_limit


    # 获取训练集dataframe
    def getDaignosisTrainData(self):
        tgtdischargetemp, tgtplatethickness, tgtplatelength2, tgttmplatetemp = eval(
            self.args["tgtdischargetemp"]), eval(self.args["tgtplatethickness"]), eval(
            self.args["tgtplatelength2"]), eval(self.args["tgttmplatetemp"]),
        SQLquery = '''
                select dd.upid,
                lmpd.productcategory,
                dd.tgtwidth,
                dd.tgtlength,
                dd.tgtthickness,
                dd.stats,
                dd.fqc_label,
                dd.toc ,
                dd.status_fqc
                from  dcenter.l2_m_primary_data lmpd
                left join dcenter.l2_m_plate lmp on lmpd.slabid = lmp.slabid
                left join dcenter.l2_cc_pdi lcp  on lmpd.slabid = lcp.slab_no
                right join app.deba_dump_data dd on dd.upid = lmp.upid
                ''' + '''
                where {tgtdischargetemp}
                and {tgtplatethickness}
                and {tgtplatelength2}
                and {tgttmplatetemp}
                and {status_cooling}
                and {fqcflag}
                ORDER BY dd.toc  DESC 
                Limit  {limit}'''.format(
            tgtdischargetemp='1=1' if tgtdischargetemp == [] else "lmpd.tgtdischargetemp > " + str(
                tgtdischargetemp[0]) + " and lmpd.tgtdischargetemp < " + str(tgtdischargetemp[1]),
            tgtplatethickness='1=1' if tgtplatethickness == [] else "(case when lmpd.shapecode ='11' or lmpd.shapecode='12' then lmpd.tgtplatethickness5 else lmpd.tgtplatethickness1 end)* 1000 > " + str(
                tgtplatethickness[
                    0]) + " and (case when lmpd.shapecode ='11' or lmpd.shapecode='12' then lmpd.tgtplatethickness5 else lmpd.tgtplatethickness1 end)* 1000 < " + str(
                tgtplatethickness[1]),
            tgtplatelength2='1=1' if tgtplatelength2 == [] else "lmpd.tgtplatelength2 > " + str(
                tgtplatelength2[0]) + " and lmpd.tgtplatelength2 < " + str(tgtplatelength2[1]),
            tgttmplatetemp='1=1' if tgttmplatetemp == [] else "lmpd.tgttmplatetemp > " + str(
                tgttmplatetemp[0]) + " and lmpd.tgttmplatetemp < " + str(tgttmplatetemp[1]),
            status_cooling='dd.status_cooling = ' + self.args['status_cooling'],
            fqcflag='dd.status_stats= 0' + (' ' if self.args['fqcflag'] == '1' else ' and dd.status_fqc= 0 '),
            limit=int(self.plate_limit)
        )
        train_rows, train_col_names = allGetSQLData(SQLquery)
        # return rows, col_names
        status_cooling, fqcflag = int(self.args["status_cooling"]), int(self.args["fqcflag"])
        return train_rows, train_col_names, status_cooling, fqcflag

    # 获取测试集dataframe
    def getDaignosisUpidData(self):
        # upids = "('21308054000', '21308055000', '21308056000', '21308057000', '21308058000', '21308059000', '21308060000')"
        SQLquery = '''
                select upid,
                platetype,                
                tgtwidth,
                tgtlength,
                tgtthickness,
                stats,
                fqc_label,
                toc,
                status_fqc
                from app.deba_dump_data 
                where {upid}
                and status_stats = 0 
                and {status_cooling}
                order by toc
        '''.format(upid='upid in ' + self.args['upids'], status_cooling='status_cooling = ' + self.args["status_cooling"])
        upid_rows, upids_col_names = allGetSQLData(SQLquery)
        return upid_rows, upids_col_names

    # and {status_stats}
    # and {status_cooling}





class GetDiAllTestDataDB:
    @staticmethod
    def getDiaTestDataDB(start_time, end_time):
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
                dd.status_cooling,
                
                lmpd.tgtdischargetemp,
                (case when lmpd.shapecode ='11' or lmpd.shapecode='12' then lmpd.tgtplatethickness5 else lmpd.tgtplatethickness1 end)* 1000 as tgtplatethickness,
                lmpd.tgtplatelength2,
                lmpd.tgttmplatetemp
                from  dcenter.l2_m_primary_data lmpd                left join dcenter.l2_m_plate lmp on lmpd.slabid = lmp.slabid
                left join dcenter.l2_cc_pdi lcp  on lmpd.slabid = lcp.slab_no
                right join app.deba_dump_data dd on dd.upid = lmp.upid
        ''' + '''
        where {start_time} 
        and {end_time}
        order by dd.toc
        '''.format(start_time='1=1' if start_time == 'all' else "dd.toc >= to_timestamp('" + str(
            start_time) + "','yyyy-mm-dd hh24:mi:ss') ",
                   end_time='1=1' if end_time == 'all' else "dd.toc <= to_timestamp('" + str(
                       end_time) + "','yyyy-mm-dd hh24:mi:ss') ",
                   )
        rows, col_names = allGetSQLData(SQLquery)
        return rows, col_names
