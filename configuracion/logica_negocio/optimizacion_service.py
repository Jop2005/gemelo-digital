import numpy as np
import pandas as pd
import os
import glob
import joblib
import warnings
from .segmentation.segmenter import Segmenter
from .weight_optimizer import WeightOptimizer
from .excepciones import SegmentacionNoEncontrada, EvaluacionNoEncontrada, ModelosNoEncontrados
from ..acceso_datos.config_repository import ConfigRepository
from ..acceso_datos.data_splitter import DataSplitter
from ..acceso_datos.config import get_config


class OptimizacionService:
    
    def __init__(self, context, model_repo):
        self._context = context
        self._model_repo = model_repo
    
    def ejecutar(self, config_loader) -> dict:
        segmenter = self._cargar_segmentador(config_loader)
        if not segmenter:
            raise SegmentacionNoEncontrada("No se encontró configuración de segmentación")
        
        evaluacion = self._cargar_evaluacion_reciente()
        if not evaluacion:
            raise EvaluacionNoEncontrada("No se encontró evaluación reciente")
        
        X_test, y_test = self._cargar_datos_prueba()
        modelos = self._model_repo.cargar_todos()
        if not modelos:
            raise ModelosNoEncontrados("No hay modelos entrenados disponibles")
        
        predicciones = {nombre: modelo.predict(X_test) for nombre, modelo in modelos.items()}
        segments = segmenter.transform(X_test)
        
        config_segmentos = self._optimizar(segmenter, modelos, predicciones, y_test, segments, evaluacion)
        return self._guardar(segmenter, config_segmentos, evaluacion)
    
    def _cargar_segmentador(self, config_loader):
        config_seg = config_loader.cargar_segmentacion()
        return Segmenter.from_config(config_seg) if config_seg else None
    
    def _cargar_evaluacion_reciente(self):
        carpeta = str(get_config().carpeta_resultados)
        eval_files = glob.glob(f'{carpeta}/evaluacion_*.joblib')
        if not eval_files:
            return None
        return joblib.load(max(eval_files, key=os.path.getctime))
    
    def _cargar_datos_prueba(self):
        loader = self._context.get_loader()
        X, y = loader.obtener_datos_completos()
        _, X_test, _, y_test = DataSplitter(test_size=0.2).split(X, y)
        return X_test, y_test
    
    def _optimizar(self, segmenter, modelos, predicciones, y_test, segments, evaluacion):
        optimizer = WeightOptimizer()
        config_segmentos = {}
        metricas = evaluacion.get('metricas', {})
        nombres = list(modelos.keys())
        for seg in range(segmenter.n_segments):
            config_segmentos[seg] = self._optimizar_segmento(seg, segments, y_test, predicciones, optimizer, metricas, nombres)
        return config_segmentos
    
    def _optimizar_segmento(self, seg, segments, y_test, predicciones, optimizer, metricas, nombres):
        mask = segments == seg
        if mask.sum() < 10:
            warnings.warn(f"Segmento {seg} tiene solo {mask.sum()} muestras.")
            return {'modelo': nombres[0]} if nombres else {'modelo': 'RandomForest'}
        y_seg = y_test.values[mask]
        pred_seg = {n: p[mask] for n, p in predicciones.items()}
        pesos = optimizer.optimizar(y_seg, pred_seg, segmento_id=seg)
        if pesos:
            pesos_filtrados = {k: v for k, v in pesos.items() if v > 0.001}
            if len(pesos_filtrados) == 1:
                return {'modelo': list(pesos_filtrados.keys())[0]}
            return {'modelo': 'Ensemble', 'pesos': {n: float(p) for n, p in pesos.items()}}
        warnings.warn(f"Optimización fallida para el segmento {seg}.")
        mejor = min(nombres, key=lambda n: float(metricas.get(n, {}).get(str(seg), {}).get('MAE', float('inf'))))
        return {'modelo': mejor}
    
    def _guardar(self, segmenter, config_segmentos, evaluacion):
        config_final = {
            'fecha': pd.Timestamp.now().isoformat(),
            'tipo_segmentacion': evaluacion.get('tipo_segmentacion'),
            'variables_segmentacion': evaluacion.get('variables_segmentacion', []),
            'segmentos': config_segmentos,
            'n_segments': segmenter.n_segments,
            'evaluacion_usada': evaluacion.get('fecha', '')
        }
        ConfigRepository().guardar_versionado(config_final, 'config_ensamblaje')
        return config_final