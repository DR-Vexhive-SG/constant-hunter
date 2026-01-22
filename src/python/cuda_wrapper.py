# cuda_wrapper.py - VERSIÓN CORREGIDA
import subprocess
import os
import sys
import json
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import shutil

@dataclass
class SearchResult:
    """Estructura para almacenar resultados de búsqueda"""
    constant_name: str
    matches: int
    time_ms: float
    throughput_gbs: float
    positions: List[int]
    description: str = ""
    digits: str = ""
    result_file: str = ""

    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            'constant_name': self.constant_name,
            'matches': self.matches,
            'time_ms': self.time_ms,
            'throughput_gbs': self.throughput_gbs,
            'positions': self.positions[:100],  # Limitar para no enviar listas enormes
            'description': self.description,
            'digits': self.digits,
            'result_file': self.result_file
        }

class CUDASearchEngine:
    """Motor de búsqueda CUDA integrado con Python - CORREGIDO"""

    def __init__(self, project_root: str = None):
        """
        Inicializa el motor de búsqueda CUDA

        Args:
            project_root: Ruta raíz del proyecto (opcional)
        """
        # Determinar ruta del proyecto de forma robusta
        if project_root is None:
            # Método 1: Desde el archivo actual
            current_dir = Path(__file__).parent
            # Ir arriba dos niveles: src/python -> src -> Constant Hunter v.1
            self.project_root = current_dir.parent.parent

            # Método alternativo: Buscar hacia arriba
            if not self.project_root.exists():
                # Intentar encontrar el proyecto desde el directorio actual de trabajo
                cwd = Path.cwd()
                possible_paths = [
                    cwd,
                    cwd.parent,
                    Path("/home/padmin/Descargas/Constant Hunter v.1"),
                    Path.home() / "Descargas" / "Constant Hunter v.1"
                ]

                for path in possible_paths:
                    bin_path = path / "bin" / "full_pi_search_enhanced"
                    if bin_path.exists():
                        self.project_root = path
                        print(f"✅ Encontrado proyecto en: {path}")
                        break
        else:
            self.project_root = Path(project_root)

        print(f"📁 Project root: {self.project_root}")  # DEBUG

        # Rutas importantes
        self.cuda_binary = self.project_root / "bin" / "full_pi_search_enhanced"
        self.results_dir = self.project_root / "results"
        self.datasets_dir = self.project_root / "datasets"

        print(f"🔍 Buscando binario en: {self.cuda_binary}")  # DEBUG

        # Verificar que exista el binario CUDA
        if not self.cuda_binary.exists():
            print(f"❌ Binario CUDA no encontrado: {self.cuda_binary}")
            print(f"   Directorio bin existe: {(self.project_root / 'bin').exists()}")
            print(f"   Contenido de bin: {list((self.project_root / 'bin').glob('*')) if (self.project_root / 'bin').exists() else 'No existe'}")

            # Intentar encontrar el binario en otras ubicaciones
            alternative_paths = [
                self.project_root / "full_pi_search_enhanced",
                Path("/home/padmin/Descargas/Constant Hunter v.1/bin/full_pi_search_enhanced"),
                Path.cwd() / "bin" / "full_pi_search_enhanced",
            ]

            for alt_path in alternative_paths:
                if alt_path.exists():
                    self.cuda_binary = alt_path
                    print(f"✅ Encontrado binario alternativo: {alt_path}")
                    break

        # Estado de la búsqueda
        self.is_running = False
        self.current_process = None
        self.latest_results_dir = None

    def search_file(
        self,
        file_path: str,
        progress_callback: Callable[[int, str], None] = None,
        output_callback: Callable[[str], None] = None
    ) -> Dict[str, SearchResult]:
        """
        Ejecuta búsqueda de constantes en un archivo - CORREGIDO
        """
        # Verificar que el archivo existe
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        print(f"🔍 Buscando 9 constantes en: {file_path.name}")  # DEBUG

        # Construir comando - usar ruta absoluta
        cmd = [str(self.cuda_binary.absolute()), str(file_path.absolute())]

        self.is_running = True
        self.latest_results_dir = None

        try:
            # Cambiar al directorio del proyecto para ejecución
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            print(f"📂 Ejecutando desde: {os.getcwd()}")  # DEBUG

            # Ejecutar proceso CUDA
            if output_callback:
                output_callback(f"🚀 Ejecutando CUDA: {' '.join(cmd)}")
                output_callback(f"📁 Directorio de trabajo: {os.getcwd()}")

            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combinar stderr con stdout
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
            )

            # Capturar salida en tiempo real
            output_lines = []
            while self.is_running:
                line = self.current_process.stdout.readline()
                if not line and self.current_process.poll() is not None:
                    break

                if line:
                    line = line.strip()
                    output_lines.append(line)

                    # Llamar a callback de salida
                    if output_callback:
                        output_callback(line)

                    # Llamar a callback de progreso si hay patrón de progreso
                    if progress_callback:
                        progress = self._parse_progress_from_line(line)
                        if progress is not None:
                            progress_callback(progress, line)

                    # Buscar directorio de resultados en la salida
                    if 'Directorio de resultados:' in line:
                        self.latest_results_dir = line.split(': ')[-1].strip()
                        print(f"📂 Directorio de resultados encontrado: {self.latest_results_dir}")  # DEBUG

            # Esperar a que termine
            return_code = self.current_process.wait()

            # Regresar al directorio original
            os.chdir(original_cwd)

            if return_code != 0:
                raise RuntimeError(f"CUDA terminó con código de error: {return_code}")

            # Si no se encontró el directorio en la salida, buscar el más reciente
            if not self.latest_results_dir:
                self.latest_results_dir = self._find_latest_results_dir()

            if not self.latest_results_dir:
                # Buscar cualquier directorio que comience con results_
                results_root = self.project_root / "results"
                if results_root.exists():
                    result_dirs = list(results_root.glob('results_*'))
                    if result_dirs:
                        # Tomar el más reciente
                        latest_dir = max(result_dirs, key=os.path.getmtime)
                        self.latest_results_dir = str(latest_dir)
                        print(f"📂 Usando directorio más reciente: {self.latest_results_dir}")  # DEBUG

            if not self.latest_results_dir:
                raise FileNotFoundError("No se pudo encontrar el directorio de resultados")

            print(f"📂 Procesando resultados de: {self.latest_results_dir}")  # DEBUG

            # Procesar resultados
            results = self._parse_results(self.latest_results_dir, output_lines)

            return results

        except Exception as e:
            # Asegurar que volvemos al directorio original
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
            print(f"❌ Error en búsqueda CUDA: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error en búsqueda CUDA: {str(e)}")

        finally:
            self.is_running = False
            self.current_process = None

    def _parse_progress_from_line(self, line: str) -> Optional[int]:
        """Parsea el progreso de una línea de salida"""
        if '[1/4] CARGANDO ARCHIVO' in line:
            return 10
        elif '[2/4] COPIANDO DATOS' in line:
            return 30
        elif '[3/4] BUSCANDO CONSTANTES' in line:
            match = re.search(r'\[(\d+)/(\d+)\]', line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                base = 50
                progress = base + int((current / total) * 40)
                return min(progress, 90)
            return 50
        elif '[4/4] FINALIZANDO' in line:
            return 95
        elif 'TIEMPO TOTAL:' in line:
            return 100

        return None

    def _find_latest_results_dir(self) -> Optional[str]:
        """Encuentra el directorio de resultados más reciente"""
        if not self.results_dir.exists():
            return None

        result_dirs = list(self.results_dir.glob('results_*'))
        if not result_dirs:
            return None

        latest_dir = max(result_dirs, key=os.path.getmtime)
        return str(latest_dir)

    def _parse_results(self, results_dir: str, output_lines: List[str]) -> Dict[str, SearchResult]:
        """Parsea los resultados de la búsqueda - MEJORADO"""
        results = {}
        results_path = Path(results_dir)

        if not results_path.exists():
            print(f"⚠️  Directorio no existe: {results_dir}")
            # Intentar con ruta relativa
            results_path = self.project_root / results_dir
            if not results_path.exists():
                raise FileNotFoundError(f"Directorio de resultados no existe: {results_dir}")

        print(f"📁 Analizando directorio: {results_path}")
        print(f"📁 Contenido: {list(results_path.glob('*'))}")

        # Buscar archivos de resultados
        result_files = list(results_path.glob('*_*.txt'))
        summary_files = list(results_path.glob('SUMMARY_*.txt'))

        print(f"📄 Archivos de resultados: {len(result_files)}")
        print(f"📄 Archivos de resumen: {len(summary_files)}")

        # Procesar cada archivo de resultados individual
        for result_file in result_files:
            if 'SUMMARY' in result_file.name:
                continue

            # Extraer nombre de constante
            const_name = result_file.name.split('_')[0]

            print(f"📖 Parseando: {result_file.name} -> {const_name}")

            try:
                const_data = self._parse_result_file(result_file)

                # Crear objeto SearchResult
                result = SearchResult(
                    constant_name=const_name,
                    matches=const_data.get('matches', 0),
                    time_ms=const_data.get('time_ms', 0),
                    throughput_gbs=const_data.get('throughput_gbs', 0),
                    positions=const_data.get('positions', []),
                    description=const_data.get('description', f'Constante {const_name}'),
                    digits=const_data.get('digits', ''),
                    result_file=str(result_file)
                )

                results[const_name] = result
                print(f"   ✅ {const_name}: {result.matches} coincidencias")

            except Exception as e:
                print(f"   ❌ Error parseando {result_file}: {e}")

        # Si no se encontraron archivos, intentar extraer de la salida
        if not results and output_lines:
            print("⚠️  No se encontraron archivos, parseando salida...")
            results = self._parse_results_from_output(output_lines)

        print(f"📊 Total de resultados parseados: {len(results)}")
        return results

    def _parse_result_file(self, result_file: Path) -> Dict[str, Any]:
        """Parsea un archivo de resultados individual"""
        data = {
            'matches': 0,
            'time_ms': 0.0,
            'throughput_gbs': 0.0,
            'positions': [],
            'description': '',
            'digits': ''
        }

        try:
            with open(result_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.split('\n')

            for i, line in enumerate(lines):
                line = line.strip()

                if line.startswith('# Descripción:'):
                    data['description'] = line.split(':', 1)[-1].strip()
                elif line.startswith('# Dígitos buscados:'):
                    digits_line = line.split(':', 1)[-1].strip()
                    # Remover espacios para obtener dígitos puros
                    data['digits'] = digits_line.replace(' ', '')
                elif line.startswith('# Coincidencias encontradas:'):
                    try:
                        data['matches'] = int(line.split(':')[-1].strip())
                    except:
                        pass
                elif line.startswith('# Tiempo de búsqueda:'):
                    try:
                        time_str = line.split(':')[-1].strip().replace(' ms', '')
                        data['time_ms'] = float(time_str)
                    except:
                        pass
                elif line.startswith('# Throughput:'):
                    try:
                        throughput_str = line.split(':')[-1].strip().replace(' GB/s', '')
                        data['throughput_gbs'] = float(throughput_str)
                    except:
                        pass
                elif line and line[0].isdigit() and not line.startswith('#'):
                    try:
                        pos = int(line.split()[0])
                        data['positions'].append(pos)
                    except:
                        pass

            print(f"   📄 {result_file.name}: {data['matches']} matches, {data['time_ms']} ms")

        except Exception as e:
            print(f"   ⚠️  Error leyendo {result_file}: {e}")

        return data

    def _parse_results_from_output(self, output_lines: List[str]) -> Dict[str, SearchResult]:
        """Intenta extraer resultados directamente de la salida"""
        results = {}

        # Patrones para buscar resultados en la salida
        for line in output_lines:
            # Buscar: "    ✓    1 coincidencias en  45.04 ms (20.68 GB/s)"
            if 'coincidencias' in line and ('✓' in line or '✗' in line):
                print(f"🔍 Analizando línea: {line}")

                # Extraer nombre de constante
                const_match = re.search(r'([chkGZ][a-z0-9]*)\s*:', line)
                if const_match:
                    const_name = const_match.group(1)

                    # Extraer número de coincidencias
                    match_match = re.search(r'(\d+)\s+coincidencias', line)
                    matches = int(match_match.group(1)) if match_match else 0

                    # Extraer tiempo
                    time_match = re.search(r'(\d+\.\d+)\s*ms', line)
                    time_ms = float(time_match.group(1)) if time_match else 0.0

                    # Extraer throughput
                    throughput_match = re.search(r'(\d+\.\d+)\s*GB/s', line)
                    throughput = float(throughput_match.group(1)) if throughput_match else 0.0

                    result = SearchResult(
                        constant_name=const_name,
                        matches=matches,
                        time_ms=time_ms,
                        throughput_gbs=throughput,
                        positions=[],
                        description=f"Constante {const_name}",
                        digits="",
                        result_file=""
                    )

                    results[const_name] = result
                    print(f"   📊 {const_name}: {matches} matches from output")

        return results

    def stop_search(self):
        """Detiene la búsqueda en curso"""
        if self.is_running and self.current_process:
            self.is_running = False
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()

    def get_available_constants(self) -> Dict[str, Dict[str, str]]:
        """Retorna la lista de constantes físicas disponibles"""
        return {
            'c': {'digits': '299792458', 'name': 'Velocidad de la luz en vacío'},
            'h': {'digits': '662607015', 'name': 'Constante de Planck'},
            'hbar': {'digits': '1054571817', 'name': 'Constante de Planck reducida'},
            'G': {'digits': '667430', 'name': 'Constante gravitacional'},
            'k': {'digits': '1380649', 'name': 'Constante de Boltzmann'},
            'mu0': {'digits': '125663706127', 'name': 'Permeabilidad magnética'},
            'epsilon0': {'digits': '88541878188', 'name': 'Permitividad eléctrica'},
            'sigma': {'digits': '5670374419', 'name': 'Constante de Stefan-Boltzmann'},
            'Z0': {'digits': '376730313412', 'name': 'Impedancia característica del vacío'}
        }

    def get_recent_results_dirs(self, limit: int = 10) -> List[Dict]:
        """Obtiene directorios de resultados recientes"""
        dirs_info = []

        if not self.results_dir.exists():
            print(f"⚠️  Directorio de resultados no existe: {self.results_dir}")
            return dirs_info

        result_dirs = sorted(
            self.results_dir.glob('results_*'),
            key=os.path.getmtime,
            reverse=True
        )[:limit]

        for dir_path in result_dirs:
            try:
                result_files = list(dir_path.glob('*_*.txt'))
                summary_files = list(dir_path.glob('SUMMARY_*.txt'))

                dir_info = {
                    'path': str(dir_path),
                    'name': dir_path.name,
                    'timestamp': dir_path.name.replace('results_', ''),
                    'result_count': len([f for f in result_files if 'SUMMARY' not in f.name]),
                    'has_summary': len(summary_files) > 0,
                    'modified': datetime.fromtimestamp(os.path.getmtime(dir_path)).strftime('%Y-%m-%d %H:%M:%S')
                }

                dirs_info.append(dir_info)

            except Exception as e:
                print(f"⚠️  Error procesando {dir_path}: {e}")
                continue

        return dirs_info
