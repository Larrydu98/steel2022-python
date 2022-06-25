import psycopg2
import json
from ..utils import readConfig, allGetSQLData

class VisualizationTsneDB:
    @staticmethod
    def getPlateYieldStaisticsData(start_time, end_time):
        SQLquery = '''
        select 
        dd.upid,
        dd.platetype,
        lmpd.steelspec,
        dd.toc,
        dd.tgtwidth,
        dd.tgtlength,
        dd.tgtthickness as tgtthickness,
        dd.stats,
        dd.fqc_label as label,
                (fqc_label->>'method1')::json->>'data' as flag_lable,
        (case when lmpd.shapecode ='11' or lmpd.shapecode='12' then lmpd.tgtplatethickness5 else lmpd.tgtplatethickness1 end) as tgtplatethickness1,
        dd.status_cooling,
        dd.status_fqc,
        lmpd.slabthickness as slabthickness,
        lmpd.tgtdischargetemp,
        lmpd.tgttmplatetemp,
        lcp.cooling_start_temp,
        lcp.cooling_stop_temp,
        lcp.cooling_rate1 
        from  dcenter.l2_m_primary_data lmpd
        left join dcenter.l2_fu_acc_t lfat on lmpd.slabid = lfat.slab_no
        left join dcenter.l2_m_plate lmp on lmpd.slabid = lmp.slabid
        left join dcenter.l2_cc_pdi lcp on lmpd.slabid = lcp.slab_no
        right join app.deba_dump_data dd on dd.upid = lmp.upid 
        ''' + '''
        where dd.status_stats= 0  
        and {start_time} 
        and {end_time} 
        order by dd.toc
        '''.format(start_time='1=1' if start_time == 'all' else "dd.toc >= to_timestamp('" + str(start_time) + "','yyyy-mm-dd hh24:mi:ss') ",
                   end_time='1=1' if end_time == 'all' else "dd.toc <= to_timestamp('" + str(end_time) + "','yyyy-mm-dd hh24:mi:ss') ",
                   )
        rows, col_names = allGetSQLData(SQLquery)
        return rows, col_names


    @staticmethod
    def getCurrentData(upids):
        current_time_sql = '''
        select 
        addd.upid, 
        addd.platetype,
        addd.tgtwidth,
        addd.tgtlength,
        addd.tgtthickness,
        addd.toc,
        addd.fqc_label,
        addd.stops,
        addd.status_fqc,
        addd.status_cooling,
        l2ff60.in_fce_time,
        l2ff60.discharge_time,
        l2ff60.staying_time_pre,
        l2ff60.staying_time_1,
        l2ff60.staying_time_2,
        l2ff60.staying_time_soak

        FROM app.deba_dump_data addd
        right join dcenter.l2_fu_flftr60 l2ff60 on addd.upid = l2ff60.upid
        where {upid} 
        order by toc     
        '''.format(upid='addd.upid in ' + upids)
        rows, col_names = allGetSQLData(current_time_sql)
        return rows, col_names