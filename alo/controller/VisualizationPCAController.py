'''
VisualizationPCAController
'''

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.decomposition import PCA
from ..api.singelSteel import data_names, without_cooling_data_names, specifications


class getVisualizationPCA:
    '''
    getVisualizationPCA
    '''

    def __init__(self):
        pass
        # print('生成实例')

    def run(self, data):
# used to fix remote data
        # read data from database which has character:
        # toc，upid，productcategory，tgtplatelength2，tgtplatethickness2，
        # tgtwidth，ave_temp_dis，crowntotal，nmrPre_params，wedgetotal，finishtemptotal，avg_p5
        # N=1000 #样本数

        # M=300 #一维变量维度

        # X = np.random.random((N,M))

        # pca = PCA(n_components=2)

        # pca.fit(X)

        # X_transformed = pca.transform(X)
        # print(X_transformed)
        # selectSql = "select * from dcenter.dump_data where upid='" + '18901034000' + "'"
        # selectSql = "select upid, toc, fqc_label from dcenter.dump_data where fqc_ismissing = 0 and toc>='2018-09-01 00:00:00' and toc<='2018-09-02 00:00:00' "
        # data = getDataBySql(selectSql)


        # path1 = os.path.abspath('.')+'/alo/PCA_data.data'

        # # fp = open(r"./PCA_data")
        # fp = open(path1)

        # allPCA = fp.readlines()
        # fp.close()

        # allPCA_df = pd.DataFrame(json.loads(allPCA[0])).T

        # allPCA_df['toc'] = pd.to_datetime(allPCA_df['toc'])

        # startTime = datetime.datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S")
        # endTime = datetime.datetime.strptime(endTime, "%Y-%m-%d %H:%M:%S")

        # somePlate_df_tmp = allPCA_df[allPCA_df['toc'] >= startTime]
        # somePlate_df = somePlate_df_tmp[somePlate_df_tmp['toc'] <= endTime]

        # somePlate_json = somePlate_df.T.to_json(orient='columns', force_ascii=False)
        # somePlate_json = json.loads(somePlate_json)

        # return somePlate_json
        # print(data)

        X = []
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
        X_embedded = PCA(n_components=2).fit_transform(X)

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