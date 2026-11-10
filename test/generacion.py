"""
Pruebas de Caja Negra para el dominio GENERACIÓN
Ejecutar con: pytest test/generacion.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# IMPORTS
# =============================================================================

# Para PREDICCIÓN
from prediccion.acceso_datos.context import DatasetContext as PrediccionContext
from prediccion.acceso_datos.config import get_config
from prediccion.acceso_datos.joblib_model_repository import JoblibModelRepository as PrediccionModelRepo
from prediccion.acceso_datos.joblib_imputer_repository import JoblibImputerRepository as PrediccionImputerRepo
from prediccion.acceso_datos.historial_repository import HistorialRepository
from prediccion.acceso_datos.file_reader import FileReader
from prediccion.acceso_datos.config_loader import ConfigLoader
from prediccion.logica_negocio.prediccion_controller import PrediccionController

# Para CONFIGURACIÓN
from configuracion.acceso_datos.context import DatasetContext as ConfigContext
from configuracion.acceso_datos.strategies_generacion import GeneracionStrategy
from configuracion.acceso_datos.joblib_model_repository import JoblibModelRepository as ConfigModelRepo

# Controladores de configuración
from configuracion.logica_negocio.segmentacion_controller import SegmentacionController
from configuracion.logica_negocio.entrenamiento_rf_controller import EntrenadorRFController
from configuracion.logica_negocio.entrenamiento_xgb_controller import EntrenadorXGBController


# =============================================================================
# CONSTANTES
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

# Columnas EXACTAS que debe tener el archivo (coinciden con dataset_generacion.json)
COLUMNAS_GENERACION = [
    'temperature_2m (C)',
    'relative_humidity_2m (%)',
    'cloud_cover (%)',
    'wind_speed_10m (km/h)',
    'shortwave_radiation (W/m2)'
]

def ruta_archivo(nombre: str) -> str:
    return str(DATA_DIR / nombre)

def archivo_existe(nombre: str) -> bool:
    return os.path.exists(ruta_archivo(nombre))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def config_generacion():
    """Configura el dominio de generación"""
    get_config().set_dominio('generacion')
    return get_config()


@pytest.fixture(scope="session")
def controller_generacion(config_generacion):
    """Crea un controlador de PREDICCIÓN para el dominio de generación."""
    try:
        from configuracion.acceso_datos.strategies_generacion import GeneracionStrategy as ConfigGenStrategy
        strategy = ConfigGenStrategy()
        context = PrediccionContext(strategy)
        
        model_repo = PrediccionModelRepo()
        imputer_repo = PrediccionImputerRepo()
        historial_repo = HistorialRepository()
        file_reader = FileReader()
        config_loader = ConfigLoader()
        
        return PrediccionController(
            context, model_repo, imputer_repo, historial_repo, file_reader, config_loader
        )
    except FileNotFoundError:
        pytest.skip("Pipeline de configuración de generación no ejecutado")
        return None


@pytest.fixture
def context_generacion_configuracion(config_generacion):
    """Contexto de CONFIGURACIÓN para el dominio de generación."""
    try:
        return ConfigContext(GeneracionStrategy())
    except FileNotFoundError:
        pytest.skip("Configuración de generación no encontrada")
        return None


@pytest.fixture
def model_repo_configuracion(config_generacion):
    """Repositorio de modelos de CONFIGURACIÓN (tiene guardar)."""
    return ConfigModelRepo()


@pytest.fixture
def features_generacion():
    """Features exactas que espera el modelo de generación."""
    return COLUMNAS_GENERACION.copy()


# =============================================================================
# FUNCIONES PARA CREAR ARCHIVOS DE PRUEBA
# =============================================================================

def crear_archivo_prueba(nombre, datos, columnas):
    """Crea un archivo CSV de prueba"""
    DATA_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(datos, columns=columnas)
    ruta = DATA_DIR / nombre
    df.to_csv(ruta, index=False)
    return ruta


def crear_datos_meteorologicos_forzado():
    """
    CP-14: Fuerza la recreación del archivo con datos meteorológicos.
    Usa EXACTAMENTE las mismas columnas que dataset_generacion.json.
    """
    archivo = DATA_DIR / 'datos_meteorologicos.csv'
    if archivo.exists():
        archivo.unlink()
    
    np.random.seed(45)
    n = 50
    
    datos = {
        'temperature_2m (C)': np.random.uniform(15, 35, n),
        'relative_humidity_2m (%)': np.random.uniform(40, 90, n),
        'cloud_cover (%)': np.random.uniform(0, 100, n),
        'wind_speed_10m (km/h)': np.random.uniform(0, 20, n),
        'shortwave_radiation (W/m2)': np.random.uniform(0, 1000, n)
    }
    
    df = pd.DataFrame(datos)
    ruta = DATA_DIR / 'datos_meteorologicos.csv'
    df.to_csv(ruta, index=False)
    return str(ruta)


# =============================================================================
# PRUEBAS DE CAJA NEGRA - GENERACIÓN
# =============================================================================

class TestGeneracion:

    # CP-08: Generar segmentación (generación)
    def test_cp08_generar_segmentacion_generacion(self, context_generacion_configuracion):
        if context_generacion_configuracion is None:
            pytest.skip("Contexto no disponible")
        controller = SegmentacionController(context_generacion_configuracion)
        segmenter = controller.ejecutar()
        assert segmenter is not None
        config = segmenter.get_config()
        assert config.get('tipo') == 'clustering'
        assert segmenter.n_segments == 6

    # CP-09: Entrenar modelos (generación)
    def test_cp09_entrenar_modelos_generacion(self, context_generacion_configuracion, model_repo_configuracion):
        if context_generacion_configuracion is None:
            pytest.skip("Contexto no disponible")
        
        rf = EntrenadorRFController(context_generacion_configuracion, model_repo_configuracion)
        xgb = EntrenadorXGBController(context_generacion_configuracion, model_repo_configuracion)
        
        resultado_rf = rf.ejecutar()
        resultado_xgb = xgb.ejecutar()
        
        assert resultado_rf is not None
        assert resultado_rf.modelo is not None
        assert resultado_rf.mae is not None
        
        assert resultado_xgb is not None
        assert resultado_xgb.modelo is not None
        assert resultado_xgb.mae is not None

    # CP-14: Predicción generación desde archivo
    def test_cp14_prediccion_generacion_archivo(self, controller_generacion, features_generacion):
        if controller_generacion is None:
            pytest.skip("Controlador no disponible")
        
        # Crear el archivo de prueba
        ruta = crear_datos_meteorologicos_forzado()
        
        # Verificar que el archivo existe
        assert os.path.exists(ruta), f"Archivo no creado: {ruta}"
        
        # Verificar que las columnas son las correctas
        df_verificacion = pd.read_csv(ruta)
        columnas_faltantes = [c for c in features_generacion if c not in df_verificacion.columns]
        assert len(columnas_faltantes) == 0, f"Columnas faltantes en el archivo: {columnas_faltantes}"
        
        # Ejecutar la predicción
        resultados = controller_generacion.predecir_archivo(
            ruta_archivo("datos_meteorologicos.csv"), features_generacion
        )
        
        # Verificar que la predicción fue exitosa (sin errores)
        assert resultados is not None
        
        # Si es un error, fallar la prueba
        if isinstance(resultados, dict) and 'error' in resultados:
            pytest.fail(f"Error al predecir: {resultados['error']}")
        
        # Verificar que se generaron predicciones
        assert len(resultados) > 0, "No se generaron predicciones"
        
        # Verificar que cada predicción tiene los campos mínimos necesarios
        for res in resultados:
            assert 'prediccion' in res, "Falta campo 'prediccion'"
            assert 'segmento' in res, "Falta campo 'segmento'"
            assert 'modelo' in res, "Falta campo 'modelo'"

    # CP-15: Predicción generación manual
    def test_cp15_prediccion_generacion_manual(self, controller_generacion):
        if controller_generacion is None:
            pytest.skip("Controlador no disponible")
        
        datos = {
            'temperature_2m (C)': 25.0,
            'relative_humidity_2m (%)': 60.0,
            'cloud_cover (%)': 50.0,
            'wind_speed_10m (km/h)': 10.0,
            'shortwave_radiation (W/m2)': 500.0
        }
        
        res = controller_generacion.predecir_manual(datos)
        
        # Verificar que la predicción fue exitosa
        assert res is not None
        assert not isinstance(res, dict) or 'error' not in res
        assert 'prediccion' in res
        assert 'segmento' in res
        assert 'modelo' in res


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])