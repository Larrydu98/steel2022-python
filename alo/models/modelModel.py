'''
模型表
'''
from . import db

class Model(db.Model):
    '''
    模型表
    '''
    __tablename__ = 'model'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    input_desc = db.Column(db.Text)
    output_desc = db.Column(db.Text)
    desc = db.Column(db.Text)
    score = db.Column(db.Float)
    step = db.Column(db.Integer)
    version = db.Column(db.String(255))
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    owner = db.Column(db.String(36, 'utf8_bin'))
    app_id = db.Column(db.String(36, 'utf8_bin'))