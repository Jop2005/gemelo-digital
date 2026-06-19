from .strategies_base import DatasetStrategy
from .data_loader import DataLoader


class ConsumoStrategy(DatasetStrategy):
    """Estrategia concreta para dataset de consumo"""
    
    def get_loader(self) -> DataLoader:
        return DataLoader.from_json('artifacts/config/consumo/dataset_consumo.json')