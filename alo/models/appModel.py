'''
应用表
'''
from . import db

class App(db.Model):
    '''
    策略表
    '''
    __tablename__ = 'app'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    version = db.Column(db.String(255))
    desc = db.Column(db.Text)
    industry = db.Column(db.String(255))
    variable_ids = db.Column(db.Text)
    quota_ids = db.Column(db.Text)
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)