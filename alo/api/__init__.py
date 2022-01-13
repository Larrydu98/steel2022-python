'''
导入文件
'''

from flask import Blueprint
from flask_restful import Api, fields

from flask_pika import Pika as FPika

api_blueprint = Blueprint('api', __name__, url_prefix='/api')

api = Api(api_blueprint)

fpika = FPika()

# from . import demo
# from . import modelTransferControllerApi
# from . import modelTransferCsvApi
from . import plateYieldStaisticsApi
# from . import Temperature2DAndFqcApi
# from . import VisualizationMaretoApi
from . import diagnosisDataApi
# from . import detailProcessApi
from . import VisualizationTsneApi
from . import VisualizationPCAApi
from . import VisualizationISOMAPApi
from . import VisualizationUMAPApi
from . import getFlag
from . import Visualization
from . import newVisualization
from . import VisualizationCorrelation
from . import VisualizationPlatetypes
from . import Platetypes
from . import singelSteel
from . import getMareyDataApi
from . import BoardNumApi
from . import newGetMareyDataApi
from . import diagnosisDataByTimeApi
from . import monitorDataByTimeApi
from . import newVisualizationByBatch
from . import getEventDataApi
# from . import modelPredictCsvAllFlagApi