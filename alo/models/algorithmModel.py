'''
算法表
'''
from . import db

class Algorithm(db.Model):
    '''
    算法表
    '''
    __tablename__ = 'algorithm'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    input_desc = db.Column(db.Text)
    output_desc = db.Column(db.Text)
    params_desc = db.Column(db.Text)
    desc = db.Column(db.Text)
    group = db.Column(db.String(255))
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    owner = db.Column(db.String(36, 'utf8_bin'))