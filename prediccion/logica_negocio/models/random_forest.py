import pandas as pd
from prediccion.logica_negocio.models.base import ModeloBase

class RandomForestModel(ModeloBase):
    def __init__(self, modelo_sklearn):
        self.modelo = modelo_sklearn
        self.nombre = "RandomForest"

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.modelo.predict(X)

    def get_nombre(self):
        return self.nombre