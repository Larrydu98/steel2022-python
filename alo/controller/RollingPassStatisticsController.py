import pandas as pd
import numpy as np
from ..api.singelSteel import filterSQL
from ..api.singelSteel import getLabelData


class RollingPassStatistics:
    def __init__(self):
        pass

    def getData(self, parser, selection, lefttable, ismissing, limitation):
        SQL, status_cooling = filterSQL(parser)
        select = ','.join(selection)
        if (SQL != ''):
            SQL += ' and ' + ismissing
        if (SQL == ''):
            SQL = "where " + ismissing
        SQL = 'select ' + select + lefttable + SQL + ' ORDER BY dd.toc  DESC Limit ' + str(limitation)
        # print(SQL)
        data, col_names = getLabelData(SQL)

        return data, col_names


    def getRollingPassStatistics(self, data):
        pass_list = []
        label_list = []
        for i, item in enumerate(data):
            pass_list.append(len(item[0]['meas']['bendingforce']))
            if item[2] == 1:
                label_list.append(404)
            else:
                if np.array(item[1]['method1']['data']).sum() == 5:  # fqc_label
                    label_list.append(1)
                else:
                    label_list.append(0)
        pass_statistics_df = pd.DataFrame({"pass": pass_list, "label":label_list}).sort_values(by=["pass", "label"])

        result = []
        for pass_no in pass_statistics_df["pass"].unique():
            pass_data = pass_statistics_df[pass_statistics_df["pass"] == pass_no]
            plate_num = len(pass_data)
            good_plate_num = len(pass_data[pass_data["label"] == 1])
            bad_plate_num = len(pass_data[pass_data["label"] == 0])
            nofqc_plate_num = len(pass_data[pass_data["label"] == 404])

            result.append({
                "pass_name": str(pass_no),
                "plate_num": plate_num,
                "good_plate_num": good_plate_num,
                "bad_plate_num": bad_plate_num,
                "nofqc_plate_num": nofqc_plate_num
            })

        return 200, result
