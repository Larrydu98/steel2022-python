import psycopg2
import json
from ..utils import readConfig


class getEventData:
    @staticmethod
    def getEventData(type, start_time, end_time):
        SQLQueryStations = '''
        lmpd.upid
        '''

        SQLQueryTimes = '''
        select
        lmpd.upid,
        lmpd.slabid,
        lmpd.steelspec,
        lmpd.productcategory,
        lmpd.slabthickness,
        lmpd.tgtdischargetemp,
        (case when lmpd.shapecode ='11' or lmpd.shapecode='12' 
        then lmpd.tgtplatethickness5 else lmpd.tgtplatethickness1 end) * 1000 as tgtplatethickness,
        lmpd.tgtwidth,
        lmpd.tgtplatelength2,
        lmpd.tgttmplatetemp,
        lmpd.adcontrolcode,
        lcp.cooling_start_temp,
        lcp.cooling_stop_temp,
        lcp.cooling_rate1,
        lmp.toc,
        
        l2ff60.in_fce_time,
        l2ff60.discharge_time,
        l2ff60.staying_time_pre,
        l2ff60.staying_time_1,
        l2ff60.staying_time_2,
        l2ff60.staying_time_soak,
        
        lmpt.pass as pass_no,
        lmpt.starttime,
        lmpt.finishtime,
        lmp.totalpassesrm,
        lmp.totalpassesfm,
        
        --lcp.start_time,
        lmp.timerollingfinish,
        lcpc.avg_time_b,
        lcpc.avg_time_w,
        
        sum(
        (case when lcps.vc_flow_top_01 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_02 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_03 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_04 != 0 then 1 else 0 end)) over (partition by lmpd.upid, lmpt.pass) as dq_count,
        
        sum(
        (case when lcps.vc_flow_top_05 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_06 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_07 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_08 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_09 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_10 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_11 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_12 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_13 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_14 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_15 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_16 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_17 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_18 != 0 then 1 else 0 end) +
        (case when lcps.vc_flow_top_19 != 0 then 1 else 0 end)) over (partition by lmpd.upid, lmpt.pass) as acc_count,
        
        l2mppr.zeropoint
        
        from (dcenter.l2_m_pass_times lmpt 
        right join dcenter.l2_fu_flftr60 l2ff60 on lmpt.upid = l2ff60.upid
        left join dcenter.l2_m_plate lmp on lmpt.upid = lmp.upid
        left join dcenter.l2_m_primary_data lmpd on lmpt.upid = lmpd.upid
        left join dcenter.l2_cc_pdi lcp on lmpt.slabid = lcp.slab_no)
        left join dcenter.l2_cc_postcalc lcpc on lmpt.slabid = lcpc.slab_no
        left join dcenter.l2_cc_preset lcps on lmpt.slabid = lcps.slab_no
        left join dcenter.l2_m_psc_pre_run l2mppr on l2mppr.slabid = lmpt.slabid and l2mppr.run = lmpt.pass
        '''

        SQLquery = (SQLQueryStations if (type=="stations") else SQLQueryTimes) + '''
            where {start_time} 
            and {end_time} 
            '''.format(start_time='1=1' if start_time == 'all' else "lmp.toc >= to_timestamp('" + str(start_time) + "','yyyy-mm-dd hh24:mi:ss') ",
                    end_time='1=1' if end_time == 'all' else "lmp.toc <= to_timestamp('" + str(end_time) + "','yyyy-mm-dd hh24:mi:ss') ",
                       ) \
                   + ('''
                   order by lmp.toc
                   '''
                      if (type == "stations") else '''
                   order by lmp.toc,lmpt.pass
                   ''')

        configArr = readConfig()
        conn = psycopg2.connect(database=configArr[0], user=configArr[1], password=configArr[2], host=configArr[3], port=configArr[4])
        # print(SQLquery)
        cursor = conn.cursor()
        cursor.execute(SQLquery)
        rows = cursor.fetchall()

        # Extract the column names
        col_names = []
        for elt in cursor.description:
            col_names.append(elt[0])

        conn.close()
        return rows, col_names

    @staticmethod
    def getMareyFlag(start_time, end_time):
        SQLQueryFQC = '''
                select
                dd.upid,
                dd.fqc_label,
                dd.status_fqc

                from app.deba_dump_data dd
                '''

        SQLquery = SQLQueryFQC + '''
                    where {start_time} 
                    and {end_time} 
                    order by dd.toc
                    '''.format(start_time='1=1' if start_time == 'all' else "dd.toc >= '" + str(start_time) + "'::timestamp ",
                               end_time='1=1' if end_time == 'all' else "dd.toc <= '" + str(end_time) + "'::timestamp ")

        configArr = readConfig()
        conn = psycopg2.connect(database=configArr[0], user=configArr[1], password=configArr[2], host=configArr[3], port=configArr[4])
        # print(SQLquery)
        cursor = conn.cursor()
        cursor.execute(SQLquery)
        rows = cursor.fetchall()

        # Extract the column names
        col_names = []
        for elt in cursor.description:
            col_names.append(elt[0])

        conn.close()
        return rows, col_names

