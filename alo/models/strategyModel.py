'''
策略表
'''
from . import db

class Strategy(db.Model):
    '''
    策略表
    '''
    __tablename__ = 'strategy'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    text = db.Column(db.String(255), nullable=False)
    input_desc = db.Column(db.Text)
    output_desc = db.Column(db.Text)
    diagram_model = db.Column(db.Text)
    model = db.Column(db.String)
    desc = db.Column(db.Text)
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    owner = db.Column(db.String(36, 'utf8_bin'))
    app_id = db.Column(db.String(36, 'utf8_bin'))
