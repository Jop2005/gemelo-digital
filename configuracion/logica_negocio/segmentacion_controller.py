from datetime import datetime
from .base_controller import BaseController
from .segmentation.segmenter import Segmenter
from configuracion.acceso_datos.config_repository import ConfigRepository


class SegmentacionController(BaseController):
    """Genera la configuración de segmentación."""
    
    def ejecutar(self):
        loader = self._get_loader()
        config_dataset = loader.config
        config_segmentacion = self._extraer_config_segmentacion(config_dataset)
    
        if config_segmentacion is None:
           return None
     
        segmenter = Segmenter()
    
        if config_segmentacion.get('tipo') == 'rules':
            segmenter.fit(None, config_segmentacion)
        else:
            df = loader.cargar_datos()
            loader.preparar_features()
            segmenter.fit(df, config_segmentacion)
        self._guardar_configuracion(segmenter)
        return segmenter
    
    def _extraer_config_segmentacion(self, config_dataset: dict) -> dict:
        """Extrae la configuración de segmentación (sin ifs)."""
        return config_dataset.get('segmentacion')
    
    def _guardar_configuracion(self, segmenter: Segmenter) -> None:
        config = segmenter.get_config()
        config['fecha'] = datetime.now().isoformat()
        
        repo = ConfigRepository()
        
        # El nombre lo da la estrategia
        nombre = segmenter._estrategia.get_nombre_archivo()
        
        repo.guardar_versionado(config, nombre)