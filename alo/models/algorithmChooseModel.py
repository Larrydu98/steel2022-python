'''
算法选择表
'''
from . import db

class AlgorithmChoose(db.Model):
    '''
    算法选择表
    '''
    __tablename__ = 'algorithm_choose'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    algorithms = db.Column(db.Text)
    model = db.Column(db.String)
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)
    model_id = db.Column(db.String(36, 'utf8_bin'))