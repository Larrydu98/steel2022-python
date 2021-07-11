'''
VisualizationMDSController
'''

import json
import numpy as np
import pandas as pd
from sklearn.manifold import MDS,Isomap
from ..api.singelSteel import data_names, without_cooling_data_names, specifications



class getVisualizationMDS:
    '''
    getVisualizationMDS
    '''

    def __init__(self):
        pass
        # print('生成实例')

    def run(self, data):
        # read data from database which has character:
        # toc，upid，productcategory，tgtplatelength2，tgtplatethickness2，tgtwidth，ave_temp_dis，
        # crowntotal，nmrPre_params，wedgetotal，finishtemptotal，avg_p5

        # N=1000 #样本数

        # M=300 #一维变量维度

        # X = np.random.random((N,M))
        # X = np.random.random((2,3))
        # print(X)
        # embedding = MDS(n_components=2)

        # X_transformed = embedding.fit_transform(X)
        # toc，upid，productcategory，tgtplatelength2，tgtplatethickness2，tgtwidth，ave_temp_dis，crowntotal，nmrPre_params，wedgetotal，finishtemptotal，avg_p5

        X = []
        X_cooling = []
        X_nocooling = []
        for item in data:
            process_data = []
            if item[9] == 0:
                for data_name in data_names:
                    process_data.append(item[6][data_name])
                X.append(process_data)
            elif item[9] == 1:
                for data_name in without_cooling_data_names:
                    process_data.append(item[6][data_name])
                X.append(process_data)

        X = pd.DataFrame(X).fillna(0).values.tolist()
        X_embedded = MDS(n_components=2).fit_transform(X)

        index = 0
        upload_json = {}
        for item in data:
            label = 0
            if item[10] == 0:
                flags = item[7]['method1']['data']
                if np.array(flags).sum() == 5:
                    label = 1
            elif item[10] == 1:
                label = 404

            single = {}
            single["x"] = X_embedded[index][0].item()
            single["y"] = X_embedded[index][1].item()
            single["toc"] = str(item[2])
            single["upid"] = item[0]
            single["label"] = str(label)
            single["status_cooling"] = item[9]
            for name in specifications:
                single[name] = item[6][name] if item[6][name] is not None else 0
            # 新增规格信息
            single["tgtthickness"] = item[5]
            single["slab_thickness"] = item[11]
            single["tgtdischargetemp"] = item[12]
            single["tgttmplatetemp"] = item[13]
            single["cooling_start_temp"] = item[14]
            single["cooling_stop_temp"] = item[15]
            single["cooling_rate1"] = item[16]

            upload_json[str(index)] = single
            index += 1
        return upload_json



