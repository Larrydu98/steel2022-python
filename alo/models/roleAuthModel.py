'''
角色权限表
'''
from . import db

class RoleAuth(db.Model):
    '''
    角色权限表
    '''
    __tablename__ = 'role_auth'

    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    role_id = db.Column(db.String(36, 'utf8_bin'), primary_key=True, nullable=False)
    auth_id = db.Column(db.String(36, 'utf8_bin'), primary_key=True, nullable=False)
