'''
样本划分表
'''
from . import db

class SampleDivide(db.Model):
    '''
    样本划分表
    '''
    __tablename__ = 'sample_divide'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    algorithm_params = db.Column(db.Text)
    trains = db.Column(db.Text)
    tests = db.Column(db.Text)
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    model_id = db.Column(db.String(36, 'utf8_bin'))
    algorithm_id = db.Column(db.String(36, 'utf8_bin'))