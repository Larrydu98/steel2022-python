'''
权限表
'''
from . import db

class Auth(db.Model):
    '''
    权限表
    '''
    __tablename__ = 'auth'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    path = db.Column(db.String(255))
    desc = db.Column(db.Text)
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)