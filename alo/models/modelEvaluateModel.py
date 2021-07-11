'''
模型评价表
'''
from . import db

class ModelEvaluate(db.Model):
    '''
    模型评价表
    '''
    __tablename__ = 'model_evaluate'

    id = db.Column(db.String(36, 'utf8_bin'), primary_key=True)
    model_id = db.Column(db.String)
    result = db.Column(db.Text)
    created_time = db.Column(db.DateTime)
    updated_time = db.Column(db.DateTime)


    def __init__(self, id, model_id, result, created_time, updated_time):
        self.id = id
        self.model_id = model_id
        self.result = result
        self.created_time = created_time
        self.updated_time = updated_time