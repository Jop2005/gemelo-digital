import os
import shutil
from pathlib import Path

ROOT = Path(r'D:\proyecto final PP')
DEST = Path(r'D:\proyecto final PP\nueva_estructura')

# Crear estructura de carpetas
for subsistema in ['configuracion', 'prediccion']:
    for capa in ['presentacion', 'logica_negocio', 'acceso_datos']:
        (DEST / subsistema / capa).mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACION
# ============================================================

# --- Presentacion ---
for f in ['console.py', 'logger.py', 'main.py']:
    src = ROOT / 'configuracion' / 'ui' / f
    dst = DEST / 'configuracion' / 'presentacion' / f
    if src.exists():
        shutil.copy2(src, dst)

# --- Logica de negocio ---
for f in ['pipeline_controller.py', 'fabricar_entrenadores.py',
          'entrenamiento_rf_controller.py', 'entrenamiento_xgb_controller.py',
          'entrenamiento_knn_controller.py', 'evaluacion_controller.py',
          'optimizacion_controller.py', 'segmentacion_controller.py',
          'interpretabilidad_controller.py']:
    src = ROOT / 'configuracion' / 'controllers' / f
    dst = DEST / 'configuracion' / 'logica_negocio' / f
    if src.exists():
        shutil.copy2(src, dst)

# Entrenamiento (subcarpeta)
src_dir = ROOT / 'configuracion' / 'controllers' / 'entrenamiento'
dst_dir = DEST / 'configuracion' / 'logica_negocio' / 'entrenamiento'
if src_dir.exists():
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

# Weight optimizer
src = ROOT / 'configuracion' / 'domain' / 'weight_optimizer.py'
dst = DEST / 'configuracion' / 'logica_negocio' / 'weight_optimizer.py'
if src.exists():
    shutil.copy2(src, dst)

# --- Acceso a datos ---
for f in ['data_loader.py', 'data_preprocessor.py', 'data_splitter.py',
          'file_reader.py', 'config_loader.py', 'config_repository.py',
          'historial_repository.py', 'joblib_model_repository.py',
          'joblib_imputer_repository.py', 'model_repository_interface.py',
          'imputer_repository_interface.py']:
    src = ROOT / 'shared' / 'persistence' / f
    dst = DEST / 'configuracion' / 'acceso_datos' / f
    if src.exists():
        shutil.copy2(src, dst)

# Config + context + strategies + utils
for f in ['config.py', 'config_loader.py']:
    src = ROOT / 'shared' / 'infrastructure' / f
    dst = DEST / 'configuracion' / 'acceso_datos' / f
    if src.exists():
        shutil.copy2(src, dst)

src = ROOT / 'shared' / 'infrastructure' / 'context.py'
dst = DEST / 'configuracion' / 'acceso_datos' / 'context.py'
if src.exists():
    shutil.copy2(src, dst)

# Strategies
src_dir = ROOT / 'shared' / 'infrastructure'
dst_dir = DEST / 'configuracion' / 'acceso_datos'
for f in ['strategies_base.py', 'strategies_consumo.py', 'strategies_generacion.py']:
    src = src_dir / f
    dst = dst_dir / f
    if src.exists():
        shutil.copy2(src, dst)

# Utils
src = ROOT / 'shared' / 'utils' / 'validators.py'
dst = DEST / 'configuracion' / 'acceso_datos' / 'validators.py'
if src.exists():
    shutil.copy2(src, dst)

# ============================================================
# PREDICCION
# ============================================================

# --- Presentacion ---
for f in ['console.py', 'logger.py', 'main.py']:
    src = ROOT / 'prediccion' / 'ui' / f
    dst = DEST / 'prediccion' / 'presentacion' / f
    if src.exists():
        shutil.copy2(src, dst)

# --- Logica de negocio ---
src = ROOT / 'prediccion' / 'controllers' / 'prediccion_controller.py'
dst = DEST / 'prediccion' / 'logica_negocio' / 'prediccion_controller.py'
if src.exists():
    shutil.copy2(src, dst)

src = ROOT / 'prediccion' / 'domain' / 'orchestrator.py'
dst = DEST / 'prediccion' / 'logica_negocio' / 'orchestrator.py'
if src.exists():
    shutil.copy2(src, dst)

# --- Acceso a datos ---
for f in ['data_loader.py', 'data_preprocessor.py', 'data_splitter.py',
          'file_reader.py', 'config_loader.py', 'config_repository.py',
          'historial_repository.py', 'joblib_model_repository.py',
          'joblib_imputer_repository.py', 'model_repository_interface.py',
          'imputer_repository_interface.py']:
    src = ROOT / 'shared' / 'persistence' / f
    dst = DEST / 'prediccion' / 'acceso_datos' / f
    if src.exists():
        shutil.copy2(src, dst)

# Config + context + strategies + utils
for f in ['config.py', 'config_loader.py']:
    src = ROOT / 'shared' / 'infrastructure' / f
    dst = DEST / 'prediccion' / 'acceso_datos' / f
    if src.exists():
        shutil.copy2(src, dst)

src = ROOT / 'shared' / 'infrastructure' / 'context.py'
dst = DEST / 'prediccion' / 'acceso_datos' / 'context.py'
if src.exists():
    shutil.copy2(src, dst)

for f in ['strategies_base.py', 'strategies_consumo.py', 'strategies_generacion.py']:
    src = ROOT / 'shared' / 'infrastructure' / f
    dst = DEST / 'prediccion' / 'acceso_datos' / f
    if src.exists():
        shutil.copy2(src, dst)

src = ROOT / 'shared' / 'utils' / 'validators.py'
dst = DEST / 'prediccion' / 'acceso_datos' / 'validators.py'
if src.exists():
    shutil.copy2(src, dst)

# ============================================================
# ARCHIVOS DUPLICADOS (evaluation, interpretability, models, segmentation, value_objects)
# ============================================================
for subsistema in ['configuracion', 'prediccion']:
    base = DEST / subsistema / 'logica_negocio'
    
    for f in ['evaluation.py', 'interpretability.py']:
        src = ROOT / 'shared' / 'domain' / f
        dst = base / f
        if src.exists():
            shutil.copy2(src, dst)
    
    for f in ['resultado_entrenamiento.py']:
        src = ROOT / 'shared' / 'domain' / 'value_objects' / f
        dst = base / f
        if src.exists():
            shutil.copy2(src, dst)
    
    # Models
    dst_models = base / 'models'
    dst_models.mkdir(exist_ok=True)
    for f in ['base.py', 'random_forest.py', 'xgboost.py']:
        src = ROOT / 'shared' / 'domain' / 'models' / f
        dst = dst_models / f
        if src.exists():
            shutil.copy2(src, dst)
    
    # Segmentation
    dst_seg = base / 'segmentation'
    dst_seg.mkdir(exist_ok=True)
    for f in ['estrategia.py', 'segmenter.py', 'reglas.py', 'percentiles.py', 'clustering.py', 'fabrica.py']:
        src = ROOT / 'shared' / 'domain' / 'segmentation' / f
        dst = dst_seg / f
        if src.exists():
            shutil.copy2(src, dst)

print("✅ Estructura creada en:", DEST)