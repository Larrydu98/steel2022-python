'''
设备表
'''
from . import db

class Equipment(db.Model):
    '''
    设备表
    '''
    __tablename__ = 'equipment'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    desc = db.Column(db.Text)
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    process_id = db.Column(db.String(36, 'utf8_bin'))
    owner = db.Column(db.String(36, 'utf8_bin'))
    app_id = db.Column(db.String(36, 'utf8_bin'))