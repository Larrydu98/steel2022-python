'''
过程表
'''
from . import db

class Quota(db.Model):
    '''
    过程表
    '''
    __tablename__ = 'quota'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    error_range = db.Column(db.Float)
    desc = db.Column(db.Text)
    type = db.Column(db.String(255))
    sample_period = db.Column(db.String(255))
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    owner = db.Column(db.String(36, 'utf8_bin'))
    app_id = db.Column(db.String(36, 'utf8_bin'))
    process_id = db.Column(db.String(36, 'utf8_bin'))
    equipment_id = db.Column(db.String(36, 'utf8_bin'))