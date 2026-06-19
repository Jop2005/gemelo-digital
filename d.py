import os
import sys

def analizar_proyecto(directorio_raiz=None):
    """Recorre el proyecto y guarda nombre y código de cada archivo .py"""
    
    if directorio_raiz is None:
        directorio_raiz = os.getcwd()
    
    # Extensiones a incluir
    extensiones = ('.py',)
    
    # Carpetas a excluir
    excluir_carpetas = {'artifacts', 'data'}
    
    # Archivos a excluir
    excluir_archivos = {'analisis_acoplamiento.py', 'd.py',  'z.py' , 'requirements-test.txt', 'requirements.txt', 'i.py'}
    
    # Lista para guardar resultados
    resultados = []
    
    print(f"🔍 Analizando proyecto en: {directorio_raiz}")
    print("=" * 80)
    
    for dirpath, dirnames, filenames in os.walk(directorio_raiz):
        # Excluir carpetas
        dirnames[:] = [d for d in dirnames if d not in excluir_carpetas]
        
        for filename in filenames:
            if filename.endswith(extensiones) and filename not in excluir_archivos:
                filepath = os.path.join(dirpath, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        contenido = f.read()
                    
                    # Ruta relativa desde la raíz
                    rel_path = os.path.relpath(filepath, directorio_raiz)
                    
                    resultados.append({
                        'path': rel_path,
                        'full_path': filepath,
                        'content': contenido
                    })
                    print(f"   ✅ {rel_path}")
                except Exception as e:
                    print(f"   ❌ Error leyendo {rel_path}: {e}")
    
    # Guardar en archivo
    output_path = os.path.join(directorio_raiz, 'reporte_completo.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"REPORTE COMPLETO DEL PROYECTO\n")
        f.write(f"Directorio: {directorio_raiz}\n")
        f.write(f"Total de archivos: {len(resultados)}\n")
        f.write("=" * 80 + "\n\n")
        
        for resultado in resultados:
            f.write("=" * 80 + "\n")
            f.write(f"📄 {resultado['path']}\n")
            f.write(f"Ruta completa: {resultado['full_path']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(resultado['content'])
            f.write("\n\n" + "=" * 80 + "\n\n")
    
    print("\n" + "=" * 80)
    print(f"✅ Reporte guardado en: {output_path}")
    print(f"📊 Total de archivos: {len(resultados)}")
    print("=" * 80)
    
    return output_path


if __name__ == "__main__":
    # Puedes cambiar la ruta si quieres analizar otra carpeta
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        ruta = os.getcwd()
    
    analizar_proyecto(ruta)