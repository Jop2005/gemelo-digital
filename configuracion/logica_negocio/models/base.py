from abc import ABC, abstractmethod


class ModeloBase(ABC):
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_nombre(self):
        pass