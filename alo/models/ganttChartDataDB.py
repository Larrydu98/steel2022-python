import psycopg2
import json
from ..utils import readConfig, allGetSQLData



class getGanttChartDataDB:
    @staticmethod
    def ganttChartDataDB(start_time, end_time):
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
                dd.status_cooling
                from  app.deba_dump_data dd 
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
