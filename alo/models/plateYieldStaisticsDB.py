import psycopg2
import json
from ..utils import readConfig, allGetSQLData

class getPlateYieldStaisticsData:
    @staticmethod
    def getPlateYieldStaisticsData(start_time, end_time):
        SQLquery = '''
        select 
        upid,toc,
        status_fqc,
        fqc_label 
        from app.deba_dump_data 
        ''' + '''
        where {start_time} 
        and {end_time} 
        '''.format(start_time='1=1' if start_time == 'all' else "toc >= to_timestamp('" + str(start_time) + "','yyyy-mm-dd hh24:mi:ss') ",
                    end_time='1=1' if end_time == 'all' else "toc <= to_timestamp('" + str(end_time) + "','yyyy-mm-dd hh24:mi:ss') ",
                       ) \
       + ('''
       order by toc
       ''')
        rows, col_names = allGetSQLData(SQLquery)
        return rows, col_names
