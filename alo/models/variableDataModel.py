'''
变量数据表
'''
from . import db

class VariableData(db.Model):
    '''
    变量数据表
    '''
    __tablename__ = 'variable_data'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    value = db.Column(db.Text)
    sample_time = db.Column(db.String(255))
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    variable_id = db.Column(db.String(36, 'utf8_bin'))
    app_id = db.Column(db.String(36, 'utf8_bin'))