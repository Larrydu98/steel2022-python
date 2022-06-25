from ..utils import readConfig, allGetSQLData
class GetInfoDataDB():
    def __init__(self, start_time, end_time):
        self.start_time, self.end_time = start_time, end_time

    def getCurrentData(self):
        current_time_sql = '''
        select 
        addd.upid, platetype,
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
        where {start_time} 
        and {end_time}
        order by toc     
        '''.format(
            start_time='1=1' if self.start_time == 'all' else "toc >= to_timestamp('" + str(self.start_time) + "','yyyy-mm-dd hh24:mi:ss') ",
            end_time='1=1' if self.end_time == 'all' else "toc <= to_timestamp('" + str(self.end_time) + "','yyyy-mm-dd hh24:mi:ss') ",
        )
        rows, col_names = allGetSQLData(current_time_sql)
        return rows, col_names

