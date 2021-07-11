'''
用户表
'''
from . import db

class User(db.Model):
    '''
    用户表
    '''
    __tablename__ = 'user'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    desc = db.Column(db.Text)
    token = db.Column(db.Text)
    pwd = db.Column(db.Text)
    tel = db.Column(db.Text)
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    app_id = db.Column(db.String(36, 'utf8_bin'))
