from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .models.base import ModeloBase

class Orchestrator:
   
    def __init__(self):
        self._segmenter = None
        self._configuracion_por_segmento: Dict[int, dict] = {}
        self._modelos: Dict[str, ModeloBase] = {}
        self._variable_segmentacion: str = None
        self._modelo_fallback_cache: Optional[str] = None
        self._nombre_modelo_fallback: Optional[str] = None
        self._metricas_evaluacion: Optional[dict] = None
    
    def initialize(
        self,
        segmenter,
        configuracion_segmentos: dict,
        modelos: Dict[str, ModeloBase],
        variable_segmentacion: str = None,
        metricas_evaluacion: dict = None
    ) -> None:
        self._segmenter = segmenter
        self._modelos = modelos
        self._configuracion_por_segmento = self._normalizar_configuracion(configuracion_segmentos)
        self._variable_segmentacion = self._determinar_variable_segmentacion(
            variable_segmentacion, segmenter
        )
        self._metricas_evaluacion = metricas_evaluacion or {}
        self._modelo_fallback_cache = None
        self._nombre_modelo_fallback = None
    
    def predict(self, muestra: pd.DataFrame) -> dict:
        self._validar_inicializacion()
        return self._predict_impl(muestra)
    
    def predict_batch(self, muestras: pd.DataFrame) -> List[dict]:
        self._validar_inicializacion()
        segmentos = self._segmenter.transform(muestras)
        resultados = [None] * len(muestras)
        indices_por_segmento = self._agrupar_por_segmento(segmentos)
        
        for segmento, indices in indices_por_segmento.items():
            indices_lista = [int(i) for i in indices]
            muestras_segmento = muestras.iloc[indices_lista].reset_index(drop=True)
            configuracion = self._obtener_configuracion(segmento)
            predicciones, incertidumbres = self._ejecutar_estrategia_batch(muestras_segmento, configuracion)
            
            for i, idx in enumerate(indices_lista):
                resultados[idx] = self._construir_resultado(
                    predicciones[i], incertidumbres[i], segmento, configuracion
                )
        
        return resultados
    
    def _validar_inicializacion(self) -> None:
        if self._segmenter is None:
            raise ValueError("Orquestador no inicializado. Llame a initialize() primero.")
    
    def _normalizar_configuracion(self, configuracion: dict) -> Dict[int, dict]:
        normalizada = {}
        for key, value in configuracion.items():
            try:
                normalizada[int(key)] = value
            except (ValueError, TypeError):
                normalizada[key] = value
        return normalizada
    
    def _determinar_variable_segmentacion(self, variable_explicita: Optional[str], segmenter) -> str:
        if variable_explicita:
            return variable_explicita
        variables = segmenter.variables if segmenter else []
        if len(variables) == 1:
            return variables[0]
        elif len(variables) > 1:
            return '+'.join(variables)
        return 'desconocida'
    
    def _obtener_segmento(self, muestra: pd.DataFrame) -> int:
        segmento_raw = self._segmenter.transform(muestra)
        return int(segmento_raw[0]) if len(segmento_raw) > 0 else -1
    
    def _obtener_configuracion(self, segmento: int) -> dict:
        return self._configuracion_por_segmento.get(segmento, {})
    
    def _agrupar_por_segmento(self, segmentos: np.ndarray) -> Dict[int, List[int]]:
        indices_por_segmento = {}
        for idx, segmento in enumerate(segmentos):
            if segmento not in indices_por_segmento:
                indices_por_segmento[segmento] = []
            indices_por_segmento[segmento].append(idx)
        return indices_por_segmento
    
    def _predict_impl(self, muestra: pd.DataFrame) -> dict:
        segmento = self._obtener_segmento(muestra)
        configuracion = self._obtener_configuracion(segmento)
        predicciones, incertidumbres = self._ejecutar_estrategia_batch(muestra, configuracion)
        return self._construir_resultado(predicciones[0], incertidumbres[0], segmento, configuracion)
    
    def _ejecutar_estrategia_batch(self, muestras: pd.DataFrame, configuracion: dict) -> Tuple[np.ndarray, np.ndarray]:
        if configuracion.get('modelo') == 'Ensemble':
            return self._ejecutar_ensamblaje(muestras, configuracion.get('pesos', {}))
        else:
            return self._ejecutar_modelo_individual(muestras, configuracion)
    
    def _ejecutar_modelo_individual(self, muestras: pd.DataFrame, configuracion: dict) -> Tuple[np.ndarray, np.ndarray]:
        nombre_modelo = configuracion.get('modelo')
        if nombre_modelo and nombre_modelo in self._modelos:
            prediccion = self._modelos[nombre_modelo].predict(muestras)
            pred = self._extraer_vector(prediccion)
            incertidumbre = np.zeros(len(pred))
            return pred, incertidumbre
        return self._ejecutar_fallback(muestras)
    
    def _ejecutar_ensamblaje(self, muestras: pd.DataFrame, pesos: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        prediccion_total = None
        suma_pesos = sum(pesos.values())
        todas_predicciones = []
        pesos_activos = []
        
        for nombre_modelo, peso in pesos.items():
            if peso > 0 and nombre_modelo in self._modelos:
                prediccion = self._modelos[nombre_modelo].predict(muestras)
                valor = self._extraer_vector(prediccion)
                todas_predicciones.append(valor)
                pesos_activos.append(peso)
                if prediccion_total is None:
                    prediccion_total = valor * peso
                else:
                    prediccion_total += valor * peso
        
        if prediccion_total is not None and suma_pesos > 0:
            pred = prediccion_total / suma_pesos
            if len(todas_predicciones) > 1:
                varianza = np.zeros(len(pred))
                for p, w in zip(todas_predicciones, pesos_activos):
                    varianza += w * (p - pred) ** 2
                incertidumbre = np.sqrt(varianza)
            else:
                incertidumbre = np.zeros(len(pred))
            return pred, incertidumbre
        
        return np.zeros(len(muestras)), np.zeros(len(muestras))
    
    def _ejecutar_fallback(self, muestras: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self._modelos:
            raise ValueError("No hay modelos disponibles para predicción")
        mejor_modelo = self._obtener_mejor_modelo_fallback()
        self._nombre_modelo_fallback = mejor_modelo
        prediccion = self._modelos[mejor_modelo].predict(muestras)
        pred = self._extraer_vector(prediccion)
        incertidumbre = np.zeros(len(pred))
        return pred, incertidumbre
    
    def _obtener_mejor_modelo_fallback(self) -> str:
        if self._modelo_fallback_cache is not None:
            return self._modelo_fallback_cache
        if self._metricas_evaluacion:
            mejor_modelo = self._seleccionar_por_mae(self._metricas_evaluacion)
            if mejor_modelo:
                self._modelo_fallback_cache = mejor_modelo
                return mejor_modelo
        self._modelo_fallback_cache = min(self._modelos.keys())
        return self._modelo_fallback_cache
    
    def _seleccionar_por_mae(self, metricas: dict) -> Optional[str]:
        mejor_modelo = None
        mejor_mae = float('inf')
        for nombre in self._modelos.keys():
            mae_global = metricas.get(nombre, {}).get('MAE', float('inf'))
            if mae_global < mejor_mae:
                mejor_mae = mae_global
                mejor_modelo = nombre
        return mejor_modelo
    
    def _extraer_vector(self, prediccion) -> np.ndarray:
        if hasattr(prediccion, '__len__') and not isinstance(prediccion, str):
            return np.array(prediccion)
        return np.array([prediccion])
    
    def _contar_modelos_activos(self, configuracion: dict) -> int:
        if configuracion.get('modelo') == 'Ensemble':
            pesos = configuracion.get('pesos', {})
            return sum(1 for p in pesos.values() if p > 0)
        return 1 if configuracion.get('modelo') else 0
    
    def _construir_resultado(self, prediccion: float, incertidumbre: float, segmento: int, configuracion: dict) -> dict:
        m = self._contar_modelos_activos(configuracion)
        if m >= 2 and incertidumbre > 0:
            margen = 1.96 * incertidumbre
            intervalo = [prediccion - margen, prediccion + margen]
        else:
            rmse = self._obtener_rmse_modelo(configuracion)
            margen = 1.96 * rmse
            intervalo = [prediccion - margen, prediccion + margen]
        
        return {
            'prediccion': float(prediccion),
            'segmento': int(segmento),
            'modelo': self._obtener_nombre_modelo(configuracion),
            'var_segmentacion': self._variable_segmentacion,
            'incertidumbre': float(incertidumbre),
            'intervalo_confianza': [float(intervalo[0]), float(intervalo[1])]
        }
    
    def _obtener_nombre_modelo(self, configuracion: dict) -> str:
        if configuracion.get('modelo') == 'Ensemble':
            pesos = configuracion.get('pesos', {})
            modelos_participantes = ', '.join(pesos.keys())
            return f"Ensemble({modelos_participantes})"
        if not configuracion.get('modelo') and self._nombre_modelo_fallback:
            return self._nombre_modelo_fallback
        return configuracion.get('modelo', 'desconocido')
    
    def _obtener_rmse_modelo(self, configuracion: dict) -> float:
        nombre = configuracion.get('modelo', '')
        if not nombre and self._nombre_modelo_fallback:
            nombre = self._nombre_modelo_fallback
        return self._metricas_evaluacion.get(nombre, {}).get('RMSE', 0.0)