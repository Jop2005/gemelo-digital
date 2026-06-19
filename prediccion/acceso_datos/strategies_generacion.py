from .strategies_base import DatasetStrategy
from .data_loader import DataLoader

class GeneracionStrategy(DatasetStrategy):
    def get_loader(self) -> DataLoader:
        return DataLoader.from_json('artifacts/config/generacion/dataset_generacion.json')