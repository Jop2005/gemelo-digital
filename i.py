import subprocess
import sys
import os
import glob
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_CONFIG = os.path.join(BASE, 'configuracion', 'presentacion', 'main.py')
SCRIPT_PREDIC = os.path.join(BASE, 'prediccion', 'presentacion', 'main.py')


def _backup(ruta):
    if not os.path.exists(ruta):
        return None
    backup = ruta + '.robustez_bak'
    if os.path.isdir(ruta):
        shutil.copytree(ruta, backup)
    else:
        shutil.copy2(ruta, backup)
    if os.path.isdir(ruta):
        shutil.rmtree(ruta)
    else:
        os.remove(ruta)
    return backup


def _restore(ruta, backup):
    if backup is None:
        return
    if os.path.isdir(backup):
        if os.path.exists(ruta):
            shutil.rmtree(ruta)
        shutil.move(backup, ruta)
    else:
        if os.path.exists(ruta):
            os.remove(ruta)
        shutil.move(backup, ruta)


def probar(nombre, dominio, script, quitar):
    print(f"\n{'='*60}")
    print(f"PRUEBA: {nombre}")
    print(f"{'='*60}")
    
    backups = {}
    for ruta in quitar:
        backups[ruta] = _backup(ruta)
    
    try:
        resultado = subprocess.run(
            [sys.executable, script],
            input=f"{dominio}\n",
            capture_output=True, text=True, timeout=30,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        salida = resultado.stdout + resultado.stderr
        
        if 'Traceback' in salida:
            print("❌ EXPLOTA (Traceback encontrado)")
            for linea in salida.split('\n'):
                if 'Error' in linea or 'File' in linea or 'traceback' in linea.lower():
                    print(f"   {linea.strip()[:150]}")
        elif 'error' in salida.lower() or 'Error' in salida:
            print("✅ MANEJADO (mensaje de error mostrado)")
            for linea in salida.split('\n'):
                if 'error' in linea.lower() or 'Error' in linea:
                    print(f"   {linea.strip()[:150]}")
        elif 'INFO' in salida or 'SUCCESS' in salida:
            print("⚠️  INESPERADO (el sistema arrancó sin errores)")
            for linea in salida.split('\n')[:5]:
                print(f"   {linea.strip()[:150]}")
        else:
            print(f"   {salida.strip()[:300]}")
    
    except subprocess.TimeoutExpired:
        print("⚠️  TIMEOUT (>30 segundos)")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    finally:
        for ruta, backup in backups.items():
            if backup:
                _restore(ruta, backup)


if __name__ == '__main__':
    
    print("=" * 60)
    print("PRUEBAS DE ROBUSTEZ - CONFIGURACION")
    print("=" * 60)
    
    probar(
        "Config: Sin dataset_consumo.json",
        "consumo", SCRIPT_CONFIG,
        [os.path.join(BASE, 'artifacts', 'config', 'consumo', 'dataset_consumo.json')]
    )
    
    probar(
        "Config: Sin dataset de entrenamiento (Excel)",
        "consumo", SCRIPT_CONFIG,
        [os.path.join(BASE, 'data', 'dataset_unificado.xlsx')]
    )
    
    seg_consumo = glob.glob(os.path.join(BASE, 'artifacts', 'config', 'consumo', 'segmentacion_*.json'))
    if seg_consumo:
        probar(
            "Config: Sin archivo de segmentacion",
            "consumo", SCRIPT_CONFIG,
            seg_consumo[:1]
        )
    
    modelos_consumo = glob.glob(os.path.join(BASE, 'artifacts', 'modelos', 'consumo', '*.joblib'))
    if modelos_consumo:
        probar(
            "Config: Sin modelos entrenados",
            "consumo", SCRIPT_CONFIG,
            [os.path.join(BASE, 'artifacts', 'modelos', 'consumo')]
        )
    
    ensamblaje_consumo = glob.glob(os.path.join(BASE, 'artifacts', 'config', 'consumo', 'config_ensamblaje_*.json'))
    if ensamblaje_consumo:
        probar(
            "Config: Sin configuracion de ensamblaje",
            "consumo", SCRIPT_CONFIG,
            ensamblaje_consumo[:1]
        )
    
    print("\n" + "=" * 60)
    print("PRUEBAS DE ROBUSTEZ - PREDICCION")
    print("=" * 60)
    
    if ensamblaje_consumo:
        probar(
            "Prediccion: Sin configuracion de ensamblaje",
            "consumo", SCRIPT_PREDIC,
            ensamblaje_consumo[:1]
        )
    
    if seg_consumo:
        probar(
            "Prediccion: Sin archivo de segmentacion",
            "consumo", SCRIPT_PREDIC,
            seg_consumo[:1]
        )
    
    if modelos_consumo:
        probar(
            "Prediccion: Sin modelos entrenados",
            "consumo", SCRIPT_PREDIC,
            [os.path.join(BASE, 'artifacts', 'modelos', 'consumo')]
        )
    
    probar(
        "Prediccion: Sin KNN Imputer",
        "consumo", SCRIPT_PREDIC,
        [os.path.join(BASE, 'artifacts', 'imputers', 'consumo')]
    )
    
    print(f"\n{'='*60}")
    print("PRUEBAS COMPLETADAS")
    print(f"{'='*60}")