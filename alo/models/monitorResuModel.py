import json
import re
from ..api.singelSteel import thicklabel
from ..utils import getLabelData


def parserMoniRequArge(parser):
    label = ["slabthickness", "tgtdischargetemp", "tgtplatethickness", "tgtwidth", "tgtplatelength2", "tgttmplatetemp", "cooling_start_temp",
             "cooling_stop_temp", "cooling_rate1", "productcategory", "steelspec", "status_cooling", "fqcflag"]
    par_label = ["steelspec", "status_cooling", "fqcflag", "toc", "tgtplatethickness"]
    for index in label:
        parser.add_argument(index, type=str, required=True)
    parser.add_argument("toc", type=str, required=True)
    args = parser.parse_args(strict=True)
    for index in par_label:
        args[index] = json.loads(args[index])

    request_body = []
    toc = args["toc"]
    del args["toc"]
    n = len(toc)
    for i in range(n):
        one_parser = {}
        for key in label:
            if key in par_label:
                one_parser[key] = json.dumps(args[key][i])
            else:
                one_parser[key] = args[key]
        request_body.append(one_parser)
    return request_body, toc


def getMonitorTrainData(args, selection, limit):
    SQL, status_cooling, fqcflag = moni_filterSQL(args)
    select = ','.join(selection)
    ismissing = ['dd.status_stats', 'dd.status_fqc']
    lefttable = ''' from  dcenter.l2_m_primary_data lmpd
                        left join dcenter.l2_m_plate lmp on lmpd.slabid = lmp.slabid
                        left join dcenter.l2_cc_pdi lcp  on lmpd.slabid = lcp.slab_no
                        right join app.deba_dump_data dd on dd.upid = lmp.upid '''
    if (SQL != ''):
        if fqcflag == 0:
            for i in ismissing:
                SQL += ' and ' + i + '= ' + '0'
        elif fqcflag == 1:
            for i in ismissing[:-1]:
                SQL += ' and ' + i + '= ' + '0'
    if (SQL == ''):
        SQL = "where " + SQL
        if fqcflag == 0:
            for i in ismissing:
                SQL += ' ' + i + '= ' + '0' + ' and '
        elif fqcflag == 1:
            for i in ismissing[:-1]:
                SQL += ' ' + i + '= ' + '0' + ' and '

        SQL = SQL[:-4]
    SQL = SQL + ' and dd.status_cooling = ' + str(status_cooling) + " "
    Limit = ' ORDER BY dd.toc  DESC Limit ' + str(limit)
    SQL = 'select ' + select + lefttable + SQL + Limit
    # print('Other Data: ', SQL)
    data, col_names = getLabelData(SQL)
    return data, col_names, status_cooling, fqcflag


def moni_filterSQL(args):
    Query = {}
    label = ["slabthickness", "tgtdischargetemp", "tgtplatethickness","tgtwidth", "tgtplatelength2", "tgttmplatetemp", "cooling_start_temp", "cooling_stop_temp", "cooling_rate1"]
    table = ["lmpd", "lmpd", "lmpd", "lmpd", "lmpd", "lmpd", "lcp", "lcp", "lcp"]
    tablename = dict(zip(label,table))
    status_cooling = args["status_cooling"]
    fqcflag = args["fqcflag"]

    for index in label:
        Query[index] = json.loads(args[index])
    SQL=''
    for index in Query:
        if(len(Query[index]) != 0):
            if index=="slabthickness" or index=="tgtwidth":
                Query[index][0]/=1000
                Query[index][1]/=1000
            if index=="tgtplatethickness":
                SQL =SQL+ thicklabel+ ' >= ' + str(Query[index][0]) +' and ' + thicklabel + ' <= ' + str(Query[index][1]) +' and '
            else:
                SQL =SQL+tablename[index]+'.' + index + ' >= ' + str(Query[index][0]) +' and ' + tablename[index] +'.' + index + ' <= ' + str(Query[index][1]) +' and '
    Query = {}
    label = ["productcategory", "steelspec"]
    table = ["lmpd","lmpd"]
    operation = dict(zip(label,[' = ', ' like ']))
    tablename = dict(zip(label,table))
    for index in label:
        Query[index] = json.loads(args[index])
    for index in Query:
        if(len(Query[index]) != 0):
            SQL=SQL+'('
            #
            for item in Query[index]:
                opera = ' = '
                if not re.search('%', item) is None:
                    opera = ' like '
                    # item = re.sub('%', "", item)
                SQL =SQL+tablename[index]+'.' + index + opera + repr(item) + ' or '
            SQL = SQL[:-4]
            SQL = SQL + ' ) and '
    SQL=SQL[:-5]
    if (SQL!=''):
        SQL="where " + SQL
    return SQL, int(status_cooling), int(fqcflag)
