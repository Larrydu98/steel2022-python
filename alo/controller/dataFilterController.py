'''
dataFilterController
'''
import pandas as pd
from flask import json


class dataFilterController:
    '''
    dataFilterController
    '''

    def __init__(self, blockedData):
        self.Fu_M_Cc_Output = blockedData['Fu_M_Cc_Output']
        self.Fu_fladcNew = blockedData['Fu_fladcNew']
        self.M_Pg_Output_new = blockedData['M_Pg_Output_new']
        self.M_Hpass_MeasNew = blockedData['M_Hpass_MeasNew']
        self.hpass_post_raw_all = blockedData['hpass_post_raw_all']
        self.raw_data_pdNew = blockedData['raw_data_pdNew']
        self.flag_df = blockedData['flag_df']

    def run(self):
        Fu_M_Cc_Output_Fu_fladc = pd.merge(self.Fu_M_Cc_Output, self.Fu_fladcNew, on='slabid')
        Fu_M_Cc_Output_Fu_fladc_drop = Fu_M_Cc_Output_Fu_fladc.dropna(axis=0, how='any')
        
        Fu_M_Cc_Output_Fu_fladc_drop = Fu_M_Cc_Output_Fu_fladc_drop.drop_duplicates('upid', 'last')
        Fu_M_Cc_Output_Fu_fladc_drop = Fu_M_Cc_Output_Fu_fladc_drop.reset_index()
        Fu_M_Cc_Output_Fu_fladc_drop = Fu_M_Cc_Output_Fu_fladc_drop.drop(['index'], axis=1)

        M_Pg_Output_new_1 = self.M_Pg_Output_new[['upid', 'centerthickness', 'leftthickness', 'rightthickness']]

        Fu_M_Cc_Output_Fu_fladc_Pg = pd.merge(Fu_M_Cc_Output_Fu_fladc_drop, M_Pg_Output_new_1,on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_drop = Fu_M_Cc_Output_Fu_fladc_Pg.dropna(axis=0, how='any')
        Fu_M_Cc_Output_Fu_fladc_Pg_drop = Fu_M_Cc_Output_Fu_fladc_Pg_drop.drop_duplicates('upid','last')
        Fu_M_Cc_Output_Fu_fladc_Pg_drop = Fu_M_Cc_Output_Fu_fladc_Pg_drop.reset_index()
        Fu_M_Cc_Output_Fu_fladc_Pg_drop = Fu_M_Cc_Output_Fu_fladc_Pg_drop.drop(['index'], axis=1)

        M_Hpass_MeasNew_1 = self.M_Hpass_MeasNew[['bendingforcebot', 'bendingforce', 'bendingforcetop', 'rollforce', 'rollforceds','rollforceos', 'screwdown',
             'shiftpos', 'speed', 'torque', 'torquebot', 'torquetop', 'upid']]

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas = pd.merge(Fu_M_Cc_Output_Fu_fladc_Pg_drop,M_Hpass_MeasNew_1, on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas.dropna(axis=0, how='any')
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_drop.drop_duplicates('upid', 'last')
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_drop.reset_index()
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_drop.drop(['index'], axis=1)

        # hpass_post_raw_all_1 = self.hpass_post_raw_all[['contactlength', 'entryflatness', 'exitflatness',
        #                                            'exitprofile', 'exittemperature',
        #                                            'exitthickness', 'exitwidth',
        #                                            'forcecorrection', 'upid']]

        hpass_post_raw_all_1 = self.hpass_post_raw_all[['contactlength', 'entryflatness', 'exitflatness',
                                                        'exitprofile', 'exittemperature','exitthickness', 'exitwidth',
                                                        'forcecorrection', 'rollforce', 'torque', 'upid']]

        hpass_post_raw_all_1.columns = ['contactlength', 'entryflatness', 'exitflatness',
                                        'exitprofile', 'exittemperature',
                                        'exitthickness', 'exitwidth',
                                        'forcecorrection', 'rollforcePost', 'torquePost', 'upid']

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost = pd.merge(Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_drop, hpass_post_raw_all_1, on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost.dropna(axis=0, how='any')
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_drop.drop_duplicates('upid', 'last')
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_drop.reset_index()
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_drop.drop(['index'], axis=1)

        raw_data_pdNew_1 = self.raw_data_pdNew[['upid', 'content']]

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent = pd.merge(Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_drop, raw_data_pdNew_1, on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent.dropna(axis=0, how='any')
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_drop.drop_duplicates('upid', 'last')
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_drop.reset_index()
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_drop.drop(['index'], axis=1)

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_Flag = pd.merge(Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_drop, self.flag_df, on='upid')

        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_Flag_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_Flag.dropna(axis=0, how='any')
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_Flag_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_Flag_drop.drop_duplicates('upid', 'last')
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_Flag_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_Flag_drop.reset_index()
        Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_Flag_drop = Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_Flag_drop.drop(['index'], axis=1)



        return Fu_M_Cc_Output_Fu_fladc_Pg_MHpassMeas_MHpassPost_CcContent_Flag_drop
