import pandas as pd
import numpy as np
from ..api.singelSteel import filterSQL
from ..utils import getLabelData


class ComputeBoardNum:

    def __init__(self, upid):
        self.upid = upid
        self.paramsIllegal = False
        try:
            # 如果以后加入参数，在这里加入
            # print('createDiagResu方法暂时没有参数')
            pass
        except Exception:
            self.paramsIllegal = True

    def getData(self, parser, selection, limit):
        SQL, status_cooling = filterSQL(parser)
        select = ','.join(selection)
        ismissing = ['dd.status_stats']
        lefttable = ''' from  dcenter.l2_m_primary_data lmpd
                            left join dcenter.l2_m_plate lmp on lmpd.slabid = lmp.slabid
                            left join dcenter.l2_cc_pdi lcp  on lmpd.slabid = lcp.slab_no
                            right join app.deba_dump_data dd on dd.upid = lmp.upid '''
        if (SQL != ''):
            for i in ismissing:
                SQL += ' and ' + i + '= ' + '0'
        if (SQL == ''):
            SQL = "where " + SQL
            for i in ismissing:
                SQL += ' ' + i + '= ' + '0' + ' and '
            SQL = SQL[:-4]
        SQL = SQL + ' and dd.status_cooling = ' + str(status_cooling) + " "
        Limit = ' ORDER BY dd.toc  DESC Limit ' + str(limit)
        SQL = 'select ' + select + lefttable + SQL + Limit
        # print(SQL)
        data, col_names = getLabelData(SQL)

        return data


    def getGoodNum(self, data):
        if len(data) == 0:
            return 204, {}

        bad_count = 0
        good_count = 0
        noFQC_count = 0
        fqc_type_list = []
        fqc_type_name = ["bend", "abnormalThickness", "horizonWave", "leftWave", "rightWave"]

        for item in data:
            if item[1] == 1:    # status_fqc
                noFQC_count += 1
            elif item[1] == 0:
                if np.array(item[2]['method1']['data']).sum() == 5:     # fqc_label
                    good_count += 1
                else:
                    bad_count += 1
                    item_fqc = item[2]['method1']['data']
                    item_fqc = list(map(lambda x: 1 if x == 0 else 0, item_fqc))
                    fqc_type_list.append(item_fqc)

        fqc_type_count = np.sum(np.array(fqc_type_list), axis=0).tolist()

        result = {}
        result["noFQC_count"] = noFQC_count
        result["good_count"] = good_count
        result["bad_count"] = bad_count
        result["total_count"] = len(data)
        fqc_type_len = len(fqc_type_list)
        for i, type in enumerate(fqc_type_name):
            if fqc_type_len == 0:
                result[type] = 0
            else:
                result[type] = fqc_type_count[i]

        return 200, result
