'''
样本选择表
'''
from . import db

class SampleChoose(db.Model):
    '''
    样本选择表
    '''
    __tablename__ = 'sample_choose'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    variable_ids = db.Column(db.Text)
    quota_ids = db.Column(db.Text)
    date_range = db.Column(db.String(255))
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    model_id = db.Column(db.String(36, 'utf8_bin'))