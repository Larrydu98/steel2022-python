'''
用户角色表
'''
from . import db

class UserRole(db.Model):
    '''
    用户角色表
    '''
    __tablename__ = 'user_role'

    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    user_id = db.Column(db.String(36, 'utf8_bin'), primary_key=True, nullable=False)
    role_id = db.Column(db.String(36, 'utf8_bin'), primary_key=True, nullable=False)