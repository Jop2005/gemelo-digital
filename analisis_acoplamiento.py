#!/usr/bin/env python3
"""
Script para calcular métricas de acoplamiento en proyectos Python
Autor: Asistente
Fecha: 2025
Versión corregida
"""

import ast
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
import csv

class CouplingAnalyzer:
    """Analizador de acoplamiento para proyectos Python"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.modules: Dict[str, Dict] = {}
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.classes: Dict[str, List[str]] = defaultdict(list)
        self.functions: Dict[str, List[str]] = defaultdict(list)
        
    def analyze(self) -> Dict:
        """Analiza todo el proyecto y retorna métricas"""
        print(f"🔍 Analizando proyecto en: {self.project_path}")
        
        # Encontrar todos los archivos Python (excluyendo este mismo script)
        python_files = [f for f in self.project_path.rglob("*.py") 
                       if f.name != "analisis_acoplamiento.py"]
        print(f"📁 Encontrados {len(python_files)} archivos Python")
        
        # Analizar cada archivo
        for file_path in python_files:
            self._analyze_file(file_path)
        
        # Calcular métricas
        metrics = self._calculate_metrics()
        
        return metrics
    
    def _analyze_file(self, file_path: Path):
        """Analiza un archivo Python individual"""
        try:
            # Leer archivo manejando BOM y diferentes codificaciones
            content = self._read_file_safe(file_path)
            if content is None:
                return
            
            tree = ast.parse(content)
            module_name = str(file_path.relative_to(self.project_path)).replace(os.sep, '.')[:-3]
            
            # Inicializar datos del módulo
            self.modules[module_name] = {
                'path': str(file_path),
                'classes': [],
                'functions': [],
                'imports': set(),
                'imported_by': set()
            }
            
            # Analizar el AST
            analyzer = ModuleAnalyzer(module_name)
            analyzer.visit(tree)
            
            # Almacenar resultados
            self.classes[module_name] = analyzer.classes
            self.functions[module_name] = analyzer.functions
            self.imports[module_name] = analyzer.imports
            
        except SyntaxError as e:
            print(f"⚠️ Error de sintaxis en {file_path}: {e}")
        except Exception as e:
            print(f"⚠️ Error analizando {file_path}: {e}")
    
    def _read_file_safe(self, file_path: Path) -> str:
        """Lee un archivo de forma segura manejando diferentes codificaciones"""
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    # Eliminar BOM si existe
                    if content and ord(content[0]) == 0xFEFF:
                        content = content[1:]
                    return content
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        print(f"⚠️ No se pudo leer {file_path} con ninguna codificación")
        return None
    
    def _calculate_metrics(self) -> Dict:
        """Calcula todas las métricas de acoplamiento"""
        metrics = {
            'project_summary': {},
            'modules': {},
            'coupling_matrix': {},
            'highly_coupled': []
        }
        
        # Primero, calcular métricas por módulo
        for module_name in self.modules:
            # Acoplamiento eferente (Ce): módulos que este módulo importa
            ce = len(self.imports.get(module_name, set()))
            
            # Acoplamiento aferente (Ca): módulos que importan este módulo
            ca = 0
            imported_by = set()
            for other_module, imports in self.imports.items():
                if other_module != module_name:
                    # Verificar si este módulo es importado por otros
                    for imp in imports:
                        if module_name in imp or imp.startswith(module_name):
                            ca += 1
                            imported_by.add(other_module)
                            break
            
            # Calcular inestabilidad (I)
            total_coupling = ce + ca
            instability = ce / total_coupling if total_coupling > 0 else 0
            
            # Nivel de acoplamiento
            if total_coupling == 0:
                coupling_level = "Sin acoplamiento"
            elif instability < 0.3:
                coupling_level = "Muy estable (bajo acoplamiento)"
            elif instability < 0.5:
                coupling_level = "Estable"
            elif instability < 0.7:
                coupling_level = "Moderadamente acoplado"
            elif instability < 0.9:
                coupling_level = "Altamente acoplado"
            else:
                coupling_level = "Extremadamente acoplado"
            
            metrics['modules'][module_name] = {
                'acoplamiento_aferente_ca': ca,
                'acoplamiento_eferente_ce': ce,
                'acoplamiento_total': total_coupling,
                'inestabilidad': round(instability, 3),
                'nivel': coupling_level,
                'clases': len(self.classes.get(module_name, [])),
                'funciones': len(self.functions.get(module_name, [])),
                'importado_por': list(imported_by),
                'importa_a': list(self.imports.get(module_name, set()))
            }
            
            # Identificar módulos altamente acoplados
            if total_coupling > 10 or (total_coupling > 0 and instability > 0.7):
                metrics['highly_coupled'].append({
                    'module': module_name,
                    'total_coupling': total_coupling,
                    'instability': round(instability, 3),
                    'reason': 'Alto acoplamiento' if total_coupling > 10 else 'Alta inestabilidad'
                })
        
        # Calcular matriz de acoplamiento
        all_modules = list(self.modules.keys())
        for module_a in all_modules:
            metrics['coupling_matrix'][module_a] = {}
            for module_b in all_modules:
                if module_a != module_b:
                    # Verificar si module_a importa module_b
                    is_imported = any(
                        module_b in imp or imp.startswith(module_b)
                        for imp in self.imports.get(module_a, set())
                    )
                    metrics['coupling_matrix'][module_a][module_b] = 1 if is_imported else 0
        
        # Calcular resumen del proyecto (después de tener todas las métricas)
        if metrics['modules']:
            all_coupling = [m['acoplamiento_total'] for m in metrics['modules'].values()]
            all_instability = [m['inestabilidad'] for m in metrics['modules'].values()]
            
            metrics['project_summary'] = {
                'total_modulos': len(self.modules),
                'modulos_altamente_acoplados': len(metrics['highly_coupled']),
                'acoplamiento_promedio': round(sum(all_coupling) / len(all_coupling), 2) if all_coupling else 0,
                'inestabilidad_promedio': round(sum(all_instability) / len(all_instability), 3) if all_instability else 0,
                'salud_arquitectonica': self._calculate_health_score(metrics)
            }
        else:
            metrics['project_summary'] = {
                'total_modulos': 0,
                'modulos_altamente_acoplados': 0,
                'acoplamiento_promedio': 0,
                'inestabilidad_promedio': 0,
                'salud_arquitectonica': "Sin módulos para evaluar"
            }
        
        return metrics
    
    def _calculate_health_score(self, metrics: Dict) -> str:
        """Calcula una puntuación de salud arquitectónica"""
        if not metrics['modules']:
            return "Sin módulos para evaluar"
        
        summary = metrics['project_summary']
        total_modules = summary.get('total_modulos', 0)
        highly_coupled = summary.get('modulos_altamente_acoplados', 0)
        avg_instability = summary.get('inestabilidad_promedio', 0)
        
        if total_modules == 0:
            return "Sin módulos para evaluar"
        
        highly_coupled_ratio = highly_coupled / total_modules if total_modules > 0 else 0
        
        if highly_coupled_ratio < 0.1 and avg_instability < 0.4:
            return "Excelente - Bajo acoplamiento"
        elif highly_coupled_ratio < 0.2 and avg_instability < 0.6:
            return "Buena - Acoplamiento moderado"
        elif highly_coupled_ratio < 0.4:
            return "Regular - Necesita mejoras"
        else:
            return "Crítica - Alto acoplamiento, requiere refactorización"

class ModuleAnalyzer(ast.NodeVisitor):
    """Visitador AST para extraer información del módulo"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.imports: Set[str] = set()
        self.classes: List[str] = []
        self.functions: List[str] = []
    
    def visit_Import(self, node: ast.Import):
        """Procesa declaraciones import"""
        for alias in node.names:
            self.imports.add(alias.name)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Procesa declaraciones from ... import"""
        if node.module:
            self.imports.add(node.module)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Procesa definiciones de clase"""
        self.classes.append(node.name)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Procesa definiciones de función"""
        self.functions.append(node.name)
        self.generic_visit(node)

def generate_reports(metrics: Dict, output_dir: str = "coupling_analysis"):
    """Genera reportes en diferentes formatos"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Reporte JSON
    json_path = output_path / "coupling_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"📊 Reporte JSON generado: {json_path}")
    
    # Reporte CSV
    csv_path = output_path / "coupling_metrics.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Módulo', 'Ca', 'Ce', 'Total', 'Inestabilidad', 'Nivel', 'Clases', 'Funciones'])
        for module, data in metrics['modules'].items():
            writer.writerow([
                module,
                data['acoplamiento_aferente_ca'],
                data['acoplamiento_eferente_ce'],
                data['acoplamiento_total'],
                data['inestabilidad'],
                data['nivel'],
                data['clases'],
                data['funciones']
            ])
    print(f"📊 Reporte CSV generado: {csv_path}")
    
    # Reporte de texto
    txt_path = output_path / "coupling_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ANÁLISIS DE ACOPLAMIENTO DEL PROYECTO\n")
        f.write("=" * 80 + "\n\n")
        
        # Resumen
        summary = metrics['project_summary']
        f.write("RESUMEN DEL PROYECTO\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total de módulos: {summary.get('total_modulos', 0)}\n")
        f.write(f"Módulos altamente acoplados: {summary.get('modulos_altamente_acoplados', 0)}\n")
        f.write(f"Acoplamiento promedio: {summary.get('acoplamiento_promedio', 0)}\n")
        f.write(f"Inestabilidad promedio: {summary.get('inestabilidad_promedio', 0)}\n")
        f.write(f"Salud arquitectónica: {summary.get('salud_arquitectonica', 'No disponible')}\n\n")
        
        # Módulos altamente acoplados
        if metrics.get('highly_coupled'):
            f.write("MÓDULOS ALTAMENTE ACOPLADOS\n")
            f.write("-" * 40 + "\n")
            for item in metrics['highly_coupled']:
                f.write(f"⚠️ {item['module']}\n")
                f.write(f"   Acoplamiento total: {item['total_coupling']}\n")
                f.write(f"   Inestabilidad: {item['instability']}\n")
                f.write(f"   Razón: {item['reason']}\n\n")
        
        # Detalle por módulo
        f.write("DETALLE POR MÓDULO\n")
        f.write("-" * 40 + "\n")
        for module, data in metrics['modules'].items():
            f.write(f"\n📦 {module}\n")
            f.write(f"   Clases: {data.get('clases', 0)}, Funciones: {data.get('funciones', 0)}\n")
            f.write(f"   Ca: {data.get('acoplamiento_aferente_ca', 0)}, Ce: {data.get('acoplamiento_eferente_ce', 0)}\n")
            f.write(f"   Total: {data.get('acoplamiento_total', 0)}, Inestabilidad: {data.get('inestabilidad', 0)}\n")
            f.write(f"   Nivel: {data.get('nivel', 'No disponible')}\n")
    
    print(f"📊 Reporte de texto generado: {txt_path}")

def main():
    """Función principal"""
    try:
        if len(sys.argv) > 1:
            project_path = sys.argv[1]
        else:
            project_path = input("📁 Ingresa la ruta del proyecto Python a analizar: ").strip()
        
        if not os.path.exists(project_path):
            print(f"❌ Error: La ruta '{project_path}' no existe")
            sys.exit(1)
        
        print("🚀 Iniciando análisis de acoplamiento...")
        print("=" * 60)
        
        analyzer = CouplingAnalyzer(project_path)
        metrics = analyzer.analyze()
        
        # Mostrar resumen en consola
        print("\n" + "=" * 60)
        print("RESUMEN DEL ANÁLISIS")
        print("=" * 60)
        summary = metrics['project_summary']
        print(f"📦 Total de módulos: {summary['total_modulos']}")
        print(f"⚠️ Módulos altamente acoplados: {summary['modulos_altamente_acoplados']}")
        print(f"📊 Acoplamiento promedio: {summary['acoplamiento_promedio']}")
        print(f"📈 Inestabilidad promedio: {summary['inestabilidad_promedio']}")
        print(f"🏥 Salud arquitectónica: {summary['salud_arquitectonica']}")
        
        # Mostrar módulos problemáticos
        if metrics['highly_coupled']:
            print(f"\n🔴 MÓDULOS ALTAMENTE ACOPLADOS:")
            for item in metrics['highly_coupled'][:5]:  # Mostrar solo los primeros 5
                print(f"   ⚠️ {item['module']} (Acoplamiento: {item['total_coupling']})")
            if len(metrics['highly_coupled']) > 5:
                print(f"   ... y {len(metrics['highly_coupled']) - 5} más")
        
        # Generar reportes
        print("\n📝 Generando reportes...")
        generate_reports(metrics)
        
        print("\n✅ Análisis completado exitosamente!")
        print("📁 Los reportes se guardaron en la carpeta 'coupling_analysis'")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Análisis interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()