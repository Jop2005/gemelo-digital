from sklearn.impute import KNNImputer
from .base_controller import BaseController
from configuracion.acceso_datos.imputer_repository_interface import ImputerRepositoryInterface

class EntrenadorKNNController(BaseController):
    
    def __init__(self, context, imputer_repo: ImputerRepositoryInterface):
       
        super().__init__(context)
        self._imputer_repo = imputer_repo
    
    def get_nombre(self) -> str:
        return "KNN Imputer"
    
    def ejecutar(self) -> KNNImputer:
      
        loader = self._get_loader()
        df = loader.obtener_features()
        
        knn = KNNImputer(n_neighbors=5)
        knn.fit(df)
        
        self._imputer_repo.guardar(knn, 'knn_imputer')
        
        return knn