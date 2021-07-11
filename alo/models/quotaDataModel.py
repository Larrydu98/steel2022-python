'''
指标数据表
'''
from . import db

class QuotaData(db.Model):
    '''
    指标数据表
    '''
    __tablename__ = 'quota_data'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    set_value = db.Column(db.Text)
    refer_value = db.Column(db.Text)
    sample_time = db.Column(db.String(255))
    actual_value = db.Column(db.Text)
    pre_value = db.Column(db.Text)
    pre_by = db.Column(db.Integer)
    refer_by = db.Column(db.Integer)
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    quota_id = db.Column(db.String(36, 'utf8_bin'))
    app_id = db.Column(db.String(36, 'utf8_bin'))