'''
数据预处理表
'''
from . import db

class DataPre(db.Model):
    '''
    数据预处理表
    '''
    __tablename__ = 'data_pre'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    algorithms = db.Column(db.Text)
    train_output = db.Column(db.String)
    test_output = db.Column(db.String)
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    model_id = db.Column(db.String(36, 'utf8_bin'))