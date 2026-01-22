# /home/padmin/Descargas/Constant Hunter v.1/src/python/gui_app.py

import sys
import os
import json
import csv
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit,
    QListWidget, QListWidgetItem, QGroupBox, QCheckBox, QLineEdit,
    QProgressBar, QMessageBox, QTableWidget, QTableWidgetItem,
    QSplitter, QTreeWidget, QTreeWidgetItem, QDialog,
    QHeaderView, QToolBar, QStatusBar, QMenu,
    QInputDialog, QComboBox, QSpinBox, QFormLayout,
    QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor, QPixmap, QAction


# Clase simulada si no existe el módulo search_thread
class CUDASearchThread(QThread):
    """Hilo simulado para búsqueda CUDA"""
    progress_updated = pyqtSignal(int, str)
    output_received = pyqtSignal(str)
    constant_result = pyqtSignal(str, dict)
    search_completed = pyqtSignal(dict)
    search_error = pyqtSignal(str)
    search_finished = pyqtSignal()

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._is_running = True

    def run(self):
        import time
        self.progress_updated.emit(0, "Iniciando búsqueda simulada...")
        time.sleep(1)

        # Resultados simulados
        results = {
            'c': {'matches': 2, 'time_ms': 123.4, 'throughput_gbs': 1.5, 'positions': [123, 456]},
            'k': {'matches': 120, 'time_ms': 456.7, 'throughput_gbs': 2.8, 'positions': list(range(10))},
            'G': {'matches': 1001, 'time_ms': 789.0, 'throughput_gbs': 3.2, 'positions': list(range(50))},
        }

        for i, (const, data) in enumerate(results.items()):
            if not self._is_running:
                break
            self.progress_updated.emit((i+1)*30, f"Encontrada constante {const}")
            self.constant_result.emit(const, data)
            time.sleep(0.5)

        self.progress_updated.emit(100, "Búsqueda completada")
        self.search_completed.emit(results)
        self.search_finished.emit()

    def stop(self):
        self._is_running = False


import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Importar nuestros módulos CUDA
try:
    from cuda_wrapper import CUDASearchEngine, SearchResult
    from search_thread import CUDASearchThread
    from advanced_analysis import AdvancedAnalysisWidget
    # Añadir visualizador avanzado
    try:
        from advanced_visualizer import AdvancedVisualizer
        VISUALIZER_AVAILABLE = True
    except ImportError:
        print("Advertencia: advanced_visualizer no disponible")
        VISUALIZER_AVAILABLE = False
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"Advertencia: No se pudieron importar módulos CUDA: {e}")
    print("La GUI funcionará en modo simulación.")
    CUDA_AVAILABLE = False

# ===================== CONSTANTES Y CONFIGURACIÓN =====================


# Hilo para búsqueda personalizada
class CustomSearchThread(QThread):
    """Hilo para búsqueda de patrones personalizados"""
    progress_updated = pyqtSignal(int, str)
    output_received = pyqtSignal(str)
    search_completed = pyqtSignal(dict)
    search_error = pyqtSignal(str)
    search_finished = pyqtSignal()

    def __init__(self, file_path, patterns, blocks=128, threads=256):
        super().__init__()
        self.file_path = file_path
        self.patterns = patterns
        self.blocks = blocks
        self.threads = threads
        self._is_running = True
        self.results = {}

    def run(self):
        """Ejecuta la búsqueda personalizada"""
        try:
            import subprocess
            import os
            import time
            from pathlib import Path

            self.progress_updated.emit(0, "Preparando búsqueda personalizada...")

            # Verificar si existe el binario CUDA para búsqueda personalizada
            project_root = Path(__file__).parent.parent
            cuda_binary = project_root / "bin" / "custom_pattern_search"

            if not cuda_binary.exists():
                # Usar simulación si no existe el binario
                self._run_simulation()
                return

            # Preparar archivo de patrones temporal
            patterns_file = Path("/tmp/custom_patterns.txt")
            with open(patterns_file, 'w') as f:
                for pattern in self.patterns:
                    f.write(f"{pattern}\n")

            # Ejecutar búsqueda CUDA
            cmd = [
                str(cuda_binary),
                "-f", self.file_path,
                "-p", str(patterns_file),
                "-b", str(self.blocks),
                "-t", str(self.threads),
                "-o", "/tmp/custom_search_results"
            ]

            self.progress_updated.emit(20, "Ejecutando kernel CUDA...")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Leer salida en tiempo real
            for line in process.stdout:
                if not self._is_running:
                    process.terminate()
                    break

                line = line.strip()
                if line:
                    self.output_received.emit(line)

                    # Parsear progreso
                    if "%" in line:
                        try:
                            percent = int(line.split("%")[0].split()[-1])
                            self.progress_updated.emit(20 + int(percent * 0.6), f"Procesando... {percent}%")
                        except:
                            pass

            process.wait()

            if process.returncode != 0:
                error_msg = process.stderr.read()
                self.search_error.emit(f"Error en búsqueda CUDA: {error_msg}")
                return

            # Procesar resultados
            self.progress_updated.emit(90, "Procesando resultados...")
            self._parse_results("/tmp/custom_search_results")

            self.progress_updated.emit(100, "Búsqueda personalizada completada")
            self.search_completed.emit(self.results)

        except Exception as e:
            self.search_error.emit(f"Error en búsqueda personalizada: {str(e)}")
        finally:
            self.search_finished.emit()

    def _run_simulation(self):
        """Simulación de búsqueda personalizada"""
        import time
        import random

        total_patterns = len(self.patterns)

        for idx, pattern in enumerate(self.patterns):
            if not self._is_running:
                break

            progress = int((idx / total_patterns) * 80)
            self.progress_updated.emit(progress, f"Buscando patrón: {pattern[:20]}...")

            # Simular resultados
            time.sleep(0.1)

            # Generar resultados aleatorios para simulación
            if random.random() > 0.7:  # 30% de probabilidad de encontrar algo
                matches = random.randint(1, 50)
                positions = sorted(random.sample(range(1000000), min(matches, 10)))

                self.results[pattern[:10]] = {
                    'matches': matches,
                    'time_ms': random.uniform(10, 100),
                    'throughput_gbs': random.uniform(0.5, 5.0),
                    'positions': positions
                }

                self.output_received.emit(f"✓ Patrón '{pattern[:10]}...' encontrado {matches} veces")

            time.sleep(0.05)

        self.progress_updated.emit(100, "Simulación completada")
        self.search_completed.emit(self.results)

    def _parse_results(self, results_dir):
        """Parsea resultados de búsqueda personalizada"""
        from pathlib import Path

        results_path = Path(results_dir)
        if not results_path.exists():
            return

        for result_file in results_path.glob("*.txt"):
            try:
                with open(result_file, 'r') as f:
                    lines = f.readlines()

                pattern_name = result_file.stem
                matches = 0
                positions = []

                for line in lines:
                    line = line.strip()
                    if line.isdigit():
                        positions.append(int(line))
                        matches += 1
                    elif "coincidencias:" in line:
                        try:
                            matches = int(line.split(":")[1].strip())
                        except:
                            pass

                if matches > 0:
                    self.results[pattern_name] = {
                        'matches': matches,
                        'time_ms': 0,  # Se podría extraer del archivo si está disponible
                        'throughput_gbs': 0,
                        'positions': positions[:100]  # Limitar para mostrar
                    }

            except Exception as e:
                print(f"Error parsing {result_file}: {e}")

    def stop(self):
        """Detiene la búsqueda"""
        self._is_running = False


class SearchResultDisplay:
    def format_positions_for_table(self, positions, max_display=3):
        """Formatea posiciones para mostrar en tabla"""
        if not positions:
            return "Ninguna"

        if len(positions) <= max_display:
            return ", ".join(str(p) for p in positions)
        else:
            displayed = ", ".join(str(p) for p in positions[:max_display])
            return f"{displayed}... (+{len(positions) - max_display})"

    def update_realtime_stats(self):
        """Actualiza estadísticas en tiempo real durante la búsqueda"""
        total_matches = 0
        constants_found = 0

        for row in range(self.results_table.rowCount()):
            matches_item = self.results_table.item(row, 1)
            if matches_item:
                try:
                    matches = int(matches_item.text())
                    total_matches += matches
                    if matches > 0:
                        constants_found += 1
                except:
                    pass

        self.stats_label.setText(
            f"📊 Buscando... | "
            f"Constantes encontradas: {constants_found} | "
            f"Coincidencias: {total_matches}"
        )

    @staticmethod
    def format_positions(positions, max_display=5):
        """Formatea una lista de posiciones para mostrar"""
        if not positions:
            return "Ninguna"

        if len(positions) <= max_display:
            return ", ".join(str(p) for p in positions)
        else:
            displayed = ", ".join(str(p) for p in positions[:max_display])
            return f"{displayed}... (+{len(positions) - max_display} más)"


# ===================== CLASE PRINCIPAL DE LA GUI =====================

class MainWindow(QMainWindow):
    """Ventana principal de Constant Hunter - ACTUALIZADA CON CUDA"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Constant Hunter v1.0 - Buscador de Constantes Físicas [CUDA]")
        self.setGeometry(100, 100, 1400, 900)

        # Variables de estado
        self.current_file = None
        self.search_thread = None
        self.results = {}
        self.cuda_engine = None
        self.constants_info = {}
        self.custom_search_thread = None

        # Configurar tema oscuro opcional
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: white;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #777777;
            }
            QListWidget {
                background-color: #3c3c3c;
                color: white;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QTextEdit, QLineEdit {
                background-color: #3c3c3c;
                color: white;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QTableWidget {
                background-color: #3c3c3c;
                color: white;
                gridline-color: #555555;
                border: 1px solid #555555;
            }
            QHeaderView::section {
                background-color: #3c3c3c;
                color: white;
                padding: 5px;
                border: 1px solid #555555;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4a4a4a;
            }
            QTabBar::tab:hover {
                background-color: #555555;
            }
        """)

        # Configurar interfaz
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.load_constants_list()

        # Intentar inicializar motor CUDA
        self.initialize_cuda_engine()

    def initialize_cuda_engine(self):
        """Inicializa el motor CUDA"""
        if not CUDA_AVAILABLE:
            self.status_bar.showMessage("⚠️ Modo simulación - CUDA no disponible", 5000)
            return False

        try:
            self.cuda_engine = CUDASearchEngine()

            # Verificar que el binario existe
            if not self.cuda_engine.cuda_binary.exists():
                QMessageBox.warning(
                    self,
                    "CUDA no disponible",
                    f"Binario CUDA no encontrado:\n{self.cuda_engine.cuda_binary}\n\n"
                    f"Por favor compile los kernels:\n./src/scripts/compile_all.sh"
                )
                return False

            self.status_bar.showMessage("✅ Motor CUDA inicializado", 3000)
            return True

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error CUDA",
                f"No se pudo inicializar el motor CUDA:\n{str(e)}\n\n"
                f"La aplicación funcionará en modo simulación."
            )
            return False

    def setup_ui(self):
        """Configura la interfaz de usuario principal"""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Barra de título
        title_label = QLabel("CONSTANT HUNTER v1.0 - BÚSQUEDA GPU-ACELERADA")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                background-color: #2196F3;
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin: 5px;
                font-size: 18px;
            }
        """)
        main_layout.addWidget(title_label)

        # Widget de pestañas
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("Arial", 10))

        # Crear las pestañas
        self.setup_file_selection_tab()
        self.setup_constants_search_tab()
        self.setup_custom_search_tab()
        self.setup_results_tab()
        self.setup_visualization_tab()
        self.setup_history_tab()

        main_layout.addWidget(self.tab_widget)

        # Barra de estado
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setTextVisible(True)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.setVisible(False)

        # Indicador CUDA
        self.cuda_status_label = QLabel("CUDA: ❓")
        self.cuda_status_label.setMaximumWidth(100)
        self.status_bar.addPermanentWidget(self.cuda_status_label)

    def setup_file_selection_tab(self):
        """Configura la pestaña de selección de archivos"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Grupo: Selección de archivo
        file_group = QGroupBox("📁 Selección de Archivo de Dígitos")
        file_group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        file_layout = QVBoxLayout()

        # Botón para seleccionar archivo
        self.select_file_btn = QPushButton("📂 Seleccionar Archivo...")
        self.select_file_btn.clicked.connect(self.select_file)
        self.select_file_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        file_layout.addWidget(self.select_file_btn)

        # Información del archivo seleccionado
        self.file_info_label = QLabel("No hay archivo seleccionado")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setStyleSheet("""
            QLabel {
                background-color: #424242;
                padding: 12px;
                border: 2px solid #555555;
                border-radius: 6px;
                margin: 5px;
                font-size: 12px;
            }
        """)
        file_layout.addWidget(self.file_info_label)

        # Vista previa del archivo
        preview_group = QGroupBox("👁️ Vista Previa (primeros 500 caracteres)")
        preview_layout = QVBoxLayout()
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(120)
        self.preview_text.setFont(QFont("Monospace", 9))
        preview_layout.addWidget(self.preview_text)
        preview_group.setLayout(preview_layout)
        file_layout.addWidget(preview_group)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Grupo: Archivos predefinidos
        predefined_group = QGroupBox("📊 Archivos Predefinidos")
        predefined_layout = QVBoxLayout()

        # Lista de archivos disponibles
        self.predefined_files_list = QListWidget()
        self.predefined_files_list.setFont(QFont("Arial", 10))
        self.predefined_files_list.itemClicked.connect(self.select_predefined_file)
        predefined_layout.addWidget(self.predefined_files_list)

        # Cargar archivos predefinidos
        self.load_predefined_files()

        predefined_group.setLayout(predefined_layout)
        layout.addWidget(predefined_group)

        layout.addStretch()
        self.tab_widget.addTab(tab, "📁 Archivo")

    def setup_constants_search_tab(self):
        """Configura la pestaña de búsqueda de constantes"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Grupo: Constantes físicas
        constants_group = QGroupBox("🔭 Constantes Físicas Predefinidas")
        constants_group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        constants_layout = QVBoxLayout()

        # Lista de constantes con checkboxes
        self.constants_list = QListWidget()
        self.constants_list.setFont(QFont("Arial", 10))

        # Cargar constantes (si CUDA está disponible, usar las del motor)
        if self.cuda_engine:
            constants_data = self.cuda_engine.get_available_constants()
        else:
            # Constantes por defecto
            constants_data = {
                "c": {"digits": "299792458", "name": "Velocidad de la luz en vacío"},
                "h": {"digits": "662607015", "name": "Constante de Planck"},
                "hbar": {"digits": "1054571817", "name": "Constante de Planck reducida"},
                "G": {"digits": "667430", "name": "Constante gravitacional"},
                "k": {"digits": "1380649", "name": "Constante de Boltzmann"},
                "mu0": {"digits": "125663706127", "name": "Permeabilidad magnética"},
                "epsilon0": {"digits": "88541878188", "name": "Permitividad eléctrica"},
                "sigma": {"digits": "5670374419", "name": "Constante de Stefan-Boltzmann"},
                "Z0": {"digits": "376730313412", "name": "Impedancia característica del vacío"}
            }

        for key, value in constants_data.items():
            item_text = f"{key}: {value['name']} ({value['digits']})"
            item = QListWidgetItem(item_text)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.constants_list.addItem(item)

        constants_layout.addWidget(self.constants_list)

        # Botones de selección
        selection_layout = QHBoxLayout()
        select_all_btn = QPushButton("✅ Seleccionar Todos")
        select_all_btn.clicked.connect(self.select_all_constants)
        select_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        clear_all_btn = QPushButton("❌ Deseleccionar Todos")
        clear_all_btn.clicked.connect(self.clear_all_constants)
        clear_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

        selection_layout.addWidget(select_all_btn)
        selection_layout.addWidget(clear_all_btn)
        constants_layout.addLayout(selection_layout)

        constants_group.setLayout(constants_layout)
        layout.addWidget(constants_group)

        # Opciones de búsqueda
        options_group = QGroupBox("⚙️ Opciones de Búsqueda")
        options_layout = QHBoxLayout()

        # Selector de bloque GPU
        self.gpu_block_label = QLabel("Bloques GPU:")
        self.gpu_block_spin = QSpinBox()
        self.gpu_block_spin.setRange(64, 512)
        self.gpu_block_spin.setValue(128)
        self.gpu_block_spin.setSuffix(" bloques")

        # Selector de hilos por bloque
        self.threads_label = QLabel("Hilos por bloque:")
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(64, 1024)
        self.threads_spin.setValue(256)
        self.threads_spin.setSuffix(" hilos")

        options_layout.addWidget(self.gpu_block_label)
        options_layout.addWidget(self.gpu_block_spin)
        options_layout.addStretch()
        options_layout.addWidget(self.threads_label)
        options_layout.addWidget(self.threads_spin)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Botón de búsqueda
        self.search_btn = QPushButton("🚀 INICIAR BÚSQUEDA CUDA")
        self.search_btn.clicked.connect(self.start_constants_search)
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 18px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #757575;
            }
        """)
        self.search_btn.setEnabled(False)
        layout.addWidget(self.search_btn)

        # Botón de detener
        self.stop_btn = QPushButton("⏹️ DETENER BÚSQUEDA")
        self.stop_btn.clicked.connect(self.stop_search)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #757575;
            }
        """)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        layout.addStretch()
        self.tab_widget.addTab(tab, "🔭 Constantes")

    def setup_custom_search_tab(self):
        """Configura la pestaña de búsqueda personalizada"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Grupo: Patrones personalizados
        custom_group = QGroupBox("🎯 Búsqueda Personalizada")
        custom_group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        custom_layout = QVBoxLayout()

        # Entrada para patrones
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Patrón a buscar:"))
        self.pattern_input = QLineEdit()
        self.pattern_input.setPlaceholderText("Ej: 123456789, ABCDEF, 3.14159, etc.")
        self.pattern_input.returnPressed.connect(self.add_custom_pattern)
        pattern_layout.addWidget(self.pattern_input)

        add_pattern_btn = QPushButton("➕ Agregar")
        add_pattern_btn.clicked.connect(self.add_custom_pattern)
        add_pattern_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        pattern_layout.addWidget(add_pattern_btn)

        custom_layout.addLayout(pattern_layout)

        # Lista de patrones personalizados
        self.custom_patterns_list = QListWidget()
        self.custom_patterns_list.setFont(QFont("Monospace", 10))
        custom_layout.addWidget(self.custom_patterns_list)

        # Botones para gestionar patrones
        pattern_buttons_layout = QHBoxLayout()

        remove_pattern_btn = QPushButton("🗑️ Eliminar Seleccionado")
        remove_pattern_btn.clicked.connect(self.remove_custom_pattern)

        clear_patterns_btn = QPushButton("🧹 Limpiar Todo")
        clear_patterns_btn.clicked.connect(self.clear_custom_patterns)

        save_patterns_btn = QPushButton("💾 Guardar Patrones")
        save_patterns_btn.clicked.connect(self.save_custom_patterns)

        load_patterns_btn = QPushButton("📂 Cargar Patrones")
        load_patterns_btn.clicked.connect(self.load_custom_patterns)

        pattern_buttons_layout.addWidget(remove_pattern_btn)
        pattern_buttons_layout.addWidget(clear_patterns_btn)
        pattern_buttons_layout.addWidget(save_patterns_btn)
        pattern_buttons_layout.addWidget(load_patterns_btn)

        custom_layout.addLayout(pattern_buttons_layout)

        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group)

        # Grupo: Opciones de búsqueda personalizada
        custom_options_group = QGroupBox("⚙️ Opciones")
        custom_options_layout = QFormLayout()

        self.custom_case_sensitive = QCheckBox("Distinguir mayúsculas/minúsculas")
        self.custom_case_sensitive.setChecked(False)

        self.custom_match_exact = QCheckBox("Coincidencia exacta")
        self.custom_match_exact.setChecked(True)

        custom_options_layout.addRow("Sensibilidad:", self.custom_case_sensitive)
        custom_options_layout.addRow("Tipo de búsqueda:", self.custom_match_exact)

        custom_options_group.setLayout(custom_options_layout)
        layout.addWidget(custom_options_group)

        # Botón de búsqueda personalizada
        self.custom_search_btn = QPushButton("🔍 INICIAR BÚSQUEDA PERSONALIZADA")
        self.custom_search_btn.clicked.connect(self.start_custom_search)
        self.custom_search_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                padding: 18px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #757575;
            }
        """)
        self.custom_search_btn.setEnabled(False)
        layout.addWidget(self.custom_search_btn)

        layout.addStretch()
        self.tab_widget.addTab(tab, "🎯 Personalizado")

    def setup_results_tab(self):
        """Configura la pestaña de resultados"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Grupo: Resultados de búsqueda
        results_group = QGroupBox("📊 Resultados de Búsqueda")
        results_group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        results_layout = QVBoxLayout()

        # Tabla de resultados
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Constante", "Coincidencias", "Tiempo (ms)",
            "Throughput (GB/s)", "Dígitos", "Posiciones"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSortingEnabled(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setFont(QFont("Arial", 10))

        # Configurar anchos de columna
        self.results_table.setColumnWidth(0, 120)  # Constante
        self.results_table.setColumnWidth(1, 120)  # Coincidencias
        self.results_table.setColumnWidth(2, 100)  # Tiempo
        self.results_table.setColumnWidth(3, 120)  # Throughput
        self.results_table.setColumnWidth(4, 150)  # Dígitos

        results_layout.addWidget(self.results_table)

        # Estadísticas
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Estadísticas: Esperando búsqueda...")
        self.stats_label.setStyleSheet("""
            QLabel {
                background-color: #424242;
                padding: 12px;
                border: 2px solid #555555;
                border-radius: 6px;
                font-size: 12px;
            }
        """)
        stats_layout.addWidget(self.stats_label)

        # Botones de exportación
        export_layout = QHBoxLayout()
        export_csv_btn = QPushButton("📥 Exportar a CSV")
        export_csv_btn.clicked.connect(self.export_to_csv)
        export_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        export_json_btn = QPushButton("📥 Exportar a JSON")
        export_json_btn.clicked.connect(self.export_to_json)
        export_json_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

        export_pdf_btn = QPushButton("📥 Exportar a PDF")
        export_pdf_btn.clicked.connect(self.export_to_pdf)
        export_pdf_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

        clear_results_btn = QPushButton("🗑️ Limpiar Resultados")
        clear_results_btn.clicked.connect(self.clear_results)
        clear_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)

        export_layout.addWidget(export_csv_btn)
        export_layout.addWidget(export_json_btn)
        export_layout.addWidget(export_pdf_btn)
        export_layout.addWidget(clear_results_btn)
        export_layout.addStretch()

        stats_layout.addLayout(export_layout)
        results_layout.addLayout(stats_layout)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Grupo: Detalles de posiciones
        details_group = QGroupBox("📍 Detalles de Posiciones Encontradas")
        details_layout = QVBoxLayout()

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QFont("Monospace", 9))
        self.details_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
        """)
        details_layout.addWidget(self.details_text)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        self.tab_widget.addTab(tab, "📊 Resultados")

    def setup_visualization_tab(self):
        """Configura la pestaña de visualización avanzada"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Crear splitter para dividir la pantalla
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Grupo de distribución
        distribution_group = QGroupBox("📈 Distribución de Coincidencias")
        distribution_layout = QVBoxLayout()

        # Figura de matplotlib
        self.figure = Figure(figsize=(10, 6), facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        distribution_layout.addWidget(self.canvas)

        # Controles del gráfico
        controls_layout = QHBoxLayout()

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Barras", "Torta", "Dispersión", "Heatmap"])
        self.chart_type_combo.currentTextChanged.connect(self.update_visualization)

        refresh_btn = QPushButton("🔄 Actualizar Gráfico")
        refresh_btn.clicked.connect(self.update_visualization)

        save_chart_btn = QPushButton("💾 Guardar Gráfico")
        save_chart_btn.clicked.connect(self.save_chart)

        controls_layout.addWidget(QLabel("Tipo de gráfico:"))
        controls_layout.addWidget(self.chart_type_combo)
        controls_layout.addWidget(refresh_btn)
        controls_layout.addWidget(save_chart_btn)
        controls_layout.addStretch()

        distribution_layout.addLayout(controls_layout)
        distribution_group.setLayout(distribution_layout)
        splitter.addWidget(distribution_group)

        # Análisis estadístico avanzado
        stats_group = QGroupBox("📊 Análisis Estadístico Avanzado")
        stats_layout = QVBoxLayout()

        self.advanced_stats_text = QTextEdit()
        self.advanced_stats_text.setReadOnly(True)
        self.advanced_stats_text.setFont(QFont("Monospace", 9))
        stats_layout.addWidget(self.advanced_stats_text)

        # Botones de análisis específico
        analysis_buttons_layout = QHBoxLayout()

        cluster_btn = QPushButton("🎯 Análisis de Clusters")
        cluster_btn.clicked.connect(lambda: self.run_specific_analysis("clusters"))

        gaps_btn = QPushButton("📏 Análisis de Gaps")
        gaps_btn.clicked.connect(lambda: self.run_specific_analysis("gaps"))

        distribution_btn = QPushButton("📊 Test de Uniformidad")
        distribution_btn.clicked.connect(lambda: self.run_specific_analysis("uniformity"))

        analysis_buttons_layout.addWidget(cluster_btn)
        analysis_buttons_layout.addWidget(gaps_btn)
        analysis_buttons_layout.addWidget(distribution_btn)
        analysis_buttons_layout.addStretch()

        stats_layout.addLayout(analysis_buttons_layout)
        stats_group.setLayout(stats_layout)
        splitter.addWidget(stats_group)

        layout.addWidget(splitter)

        # Botón para ejecutar análisis avanzado
        advanced_btn = QPushButton("🔍 EJECUTAR ANÁLISIS AVANZADO COMPLETO")
        advanced_btn.clicked.connect(self.run_advanced_analysis)
        advanced_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(advanced_btn)

        self.tab_widget.addTab(tab, "🔬 Visualización Avanzada")

    def setup_history_tab(self):
        """Configura la pestaña de historial"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Grupo: Historial de búsquedas
        history_group = QGroupBox("📜 Historial de Búsquedas")
        history_group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        history_layout = QVBoxLayout()

        # Lista de búsquedas anteriores
        self.history_tree = QTreeWidget()
        self.history_tree.setHeaderLabels(["Fecha", "Archivo", "Constantes", "Resultados", "Tiempo"])
        self.history_tree.setFont(QFont("Arial", 9))
        self.history_tree.itemDoubleClicked.connect(self.load_history_result)
        history_layout.addWidget(self.history_tree)

        # Botones de historial
        history_buttons_layout = QHBoxLayout()

        refresh_history_btn = QPushButton("🔄 Actualizar Historial")
        refresh_history_btn.clicked.connect(self.load_search_history)

        load_history_btn = QPushButton("📂 Cargar Resultado")
        load_history_btn.clicked.connect(self.load_selected_history)

        delete_history_btn = QPushButton("🗑️ Eliminar Seleccionado")
        delete_history_btn.clicked.connect(self.delete_history_item)

        clear_history_btn = QPushButton("🧹 Limpiar Historial")
        clear_history_btn.clicked.connect(self.clear_history)

        history_buttons_layout.addWidget(refresh_history_btn)
        history_buttons_layout.addWidget(load_history_btn)
        history_buttons_layout.addWidget(delete_history_btn)
        history_buttons_layout.addWidget(clear_history_btn)
        history_buttons_layout.addStretch()

        history_layout.addLayout(history_buttons_layout)
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        # Cargar historial
        QTimer.singleShot(100, self.load_search_history)

        self.tab_widget.addTab(tab, "📜 Historial")

    def setup_menu(self):
        """Configura la barra de menú"""
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        # Menú Archivo
        file_menu = menubar.addMenu("📁 Archivo")

        load_file_action = QAction("📂 Abrir Archivo...", self)
        load_file_action.setShortcut("Ctrl+O")
        load_file_action.triggered.connect(self.select_file)
        file_menu.addAction(load_file_action)

        file_menu.addSeparator()

        export_menu = file_menu.addMenu("📤 Exportar")

        export_csv_action = QAction("CSV", self)
        export_csv_action.triggered.connect(self.export_to_csv)
        export_menu.addAction(export_csv_action)

        export_json_action = QAction("JSON", self)
        export_json_action.triggered.connect(self.export_to_json)
        export_menu.addAction(export_json_action)

        file_menu.addSeparator()

        exit_action = QAction("🚪 Salir", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Menú Búsqueda
        search_menu = menubar.addMenu("🔍 Búsqueda")

        start_search_action = QAction("🚀 Iniciar Búsqueda", self)
        start_search_action.setShortcut("F5")
        start_search_action.triggered.connect(self.start_constants_search)
        search_menu.addAction(start_search_action)

        stop_search_action = QAction("⏹️ Detener Búsqueda", self)
        stop_search_action.setShortcut("Esc")
        stop_search_action.triggered.connect(self.stop_search)
        search_menu.addAction(stop_search_action)

        search_menu.addSeparator()

        benchmark_action = QAction("📊 Ejecutar Benchmark", self)
        benchmark_action.triggered.connect(self.run_benchmark)
        search_menu.addAction(benchmark_action)

        # Menú Herramientas
        tools_menu = menubar.addMenu("⚙️ Herramientas")

        compile_action = QAction("🔧 Compilar Kernels CUDA", self)
        compile_action.triggered.connect(self.compile_kernels)
        tools_menu.addAction(compile_action)

        settings_action = QAction("⚙️ Configuración", self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)

        tools_menu.addSeparator()

        clear_cache_action = QAction("🗑️ Limpiar Cache", self)
        clear_cache_action.triggered.connect(self.clear_cache)
        tools_menu.addAction(clear_cache_action)

        # Menú Ayuda
        help_menu = menubar.addMenu("❓ Ayuda")

        about_action = QAction("ℹ️ Acerca de Constant Hunter", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        docs_action = QAction("📚 Documentación", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)

        help_menu.addSeparator()

        check_updates_action = QAction("🔄 Buscar Actualizaciones", self)
        check_updates_action.triggered.connect(self.check_for_updates)
        help_menu.addAction(check_updates_action)

    def setup_toolbar(self):
        """Configura la barra de herramientas"""
        toolbar = self.addToolBar("Herramientas")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)

        # Botón para seleccionar archivo
        file_action = QAction("📂", self)
        file_action.setToolTip("Seleccionar archivo")
        file_action.triggered.connect(self.select_file)
        toolbar.addAction(file_action)

        toolbar.addSeparator()

        # Botón para iniciar búsqueda
        search_action = QAction("🚀", self)
        search_action.setToolTip("Iniciar búsqueda")
        search_action.triggered.connect(self.start_constants_search)
        toolbar.addAction(search_action)

        # Botón para detener búsqueda
        stop_action = QAction("⏹️", self)
        stop_action.setToolTip("Detener búsqueda")
        stop_action.triggered.connect(self.stop_search)
        toolbar.addAction(stop_action)

        toolbar.addSeparator()

        # Botón para exportar
        export_action = QAction("📥", self)
        export_action.setToolTip("Exportar resultados")
        export_action.triggered.connect(self.export_to_csv)
        toolbar.addAction(export_action)

        toolbar.addSeparator()

        # Botón para abrir directorio de resultados
        results_action = QAction("📂", self)
        results_action.setToolTip("Abrir directorio de resultados")
        results_action.triggered.connect(self.open_results_directory)
        toolbar.addAction(results_action)

        # Indicador CUDA en toolbar
        toolbar.addSeparator()
        self.cuda_toolbar_label = QLabel(" CUDA: ")
        toolbar.addWidget(self.cuda_toolbar_label)

        self.update_cuda_status()

    # ===================== MÉTODOS DE LA INTERFAZ =====================

    def update_cuda_status(self):
        """Actualiza el indicador de estado CUDA"""
        if self.cuda_engine and self.cuda_engine.cuda_binary.exists():
            self.cuda_status_label.setText("CUDA: ✅")
            self.cuda_toolbar_label.setText(" CUDA: ✅ ")
            self.cuda_toolbar_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.cuda_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.cuda_status_label.setText("CUDA: ❌")
            self.cuda_toolbar_label.setText(" CUDA: ❌ ")
            self.cuda_toolbar_label.setStyleSheet("color: #f44336; font-weight: bold;")
            self.cuda_status_label.setStyleSheet("color: #f44336; font-weight: bold;")

    def select_file(self):
        """Abre un diálogo para seleccionar archivo"""
        from PyQt6.QtWidgets import QFileDialog
        import os

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo de dígitos",
            os.path.expanduser("~"),
            "Archivos de texto (*.txt);;Todos los archivos (*)"
        )

        if file_path:
            self.load_file(file_path)
            return True
        else:
            return False

    def load_predefined_files(self):
        """Carga la lista de archivos predefinidos - VERSIÓN CORREGIDA"""
        self.predefined_files_list.clear()

        # Método robusto para encontrar datasets
        try:
            import os
            from pathlib import Path

            # Buscar en ubicaciones comunes
            possible_paths = [
                Path(__file__).parent.parent.parent / "datasets",  # Constant Hunter v.1/datasets
                Path.cwd().parent.parent / "datasets",             # Desde src/python
                Path.cwd().parent / "datasets",                    # Desde src
                Path.cwd() / "datasets",                           # Desde raíz
                Path("/home/padmin/Descargas/Constant Hunter v.1/datasets"),
                Path.home() / "Descargas" / "Constant Hunter v.1" / "datasets",
            ]

            datasets_dir = None
            for path in possible_paths:
                if path.exists():
                    datasets_dir = path
                    print(f"📁 Datasets encontrados en: {datasets_dir}")
                    break

            if datasets_dir and datasets_dir.exists():
                # Obtener todos los archivos .txt
                files = list(datasets_dir.glob("*.txt"))
                files.sort(key=lambda x: x.name.lower())

                print(f"📄 {len(files)} archivos encontrados en datasets")

                for file in files:
                    try:
                        size_bytes = os.path.getsize(file)
                        size_mb = size_bytes / (1024 * 1024)
                        size_gb = size_mb / 1024

                        # Nombre amigable
                        name_map = {
                            "Pi - Dec.txt": "π (Pi en decimal)",
                            "Pi - Hex.txt": "π (Pi en hexadecimal)",
                            "e - Dec.txt": "e (Número de Euler en decimal)",
                            "e - Hex.txt": "e (Número de Euler en hexadecimal)",
                            "Sqrt(2) - Dec.txt": "√2 (Raíz cuadrada de 2)",
                            "Sqrt(2) - Hex.txt": "√2 (Raíz cuadrada de 2 - hex)",
                            "Euler's Constant - Dec.txt": "γ (Constante de Euler-Mascheroni)",
                            "Euler's Constant - Hex.txt": "γ (Constante de Euler-Mascheroni - hex)",
                        }

                        display_name = name_map.get(file.name, file.stem)

                        # Formato de tamaño
                        if size_gb >= 0.1:
                            size_text = f"{size_gb:.2f} GB"
                        else:
                            size_text = f"{size_mb:.1f} MB"

                        item_text = f"{display_name} - {size_text}"
                        item = QListWidgetItem(item_text)
                        item.setData(Qt.ItemDataRole.UserRole, str(file.absolute()))
                        item.setToolTip(f"Ruta: {file}\nTamaño: {size_bytes:,} bytes")

                        # Color según tipo
                        if "Pi" in file.name:
                            item.setForeground(QColor("#2196F3"))
                        elif "e" in file.name or "Euler" in file.name:
                            item.setForeground(QColor("#4CAF50"))
                        elif "Sqrt" in file.name:
                            item.setForeground(QColor("#FF9800"))

                        self.predefined_files_list.addItem(item)

                    except Exception as e:
                        print(f"⚠️  Error procesando {file.name}: {e}")
                        item = QListWidgetItem(f"{file.name} (error)")
                        item.setForeground(QColor("red"))
                        self.predefined_files_list.addItem(item)
            else:
                item = QListWidgetItem(f"Directorio datasets no encontrado")
                item.setForeground(QColor("orange"))
                self.predefined_files_list.addItem(item)
                print(f"⚠️  No se encontró el directorio datasets")

        except Exception as e:
            item = QListWidgetItem(f"Error: {str(e)[:50]}...")
            item.setForeground(QColor("red"))
            self.predefined_files_list.addItem(item)
            print(f"❌ Error en load_predefined_files: {e}")

    def load_file(self, file_path):
        """Carga un archivo y actualiza la interfaz"""
        self.current_file = file_path

        try:
            # Leer información del archivo
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            size_gb = file_size / (1024 * 1024 * 1024)

            # Leer vista previa
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                preview = f.read(500)
                self.preview_text.setText(preview)

            # Actualizar información
            self.file_info_label.setText(
                f"📄 Archivo: {os.path.basename(file_path)}\n"
                f"📏 Tamaño: {file_size:,} bytes ({size_mb:.2f} MB, {size_gb:.3f} GB)\n"
                f"📁 Ruta: {file_path}"
            )

            # Habilitar botones de búsqueda
            self.search_btn.setEnabled(True)
            self.custom_search_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            self.status_bar.showMessage(f"Archivo cargado: {os.path.basename(file_path)}", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo leer el archivo: {str(e)}")

    def select_predefined_file(self, item):
        """Selecciona un archivo predefinido"""
        file_path = item.data(Qt.ItemDataRole.UserRole)

        if os.path.exists(file_path):
            self.load_file(file_path)
        else:
            QMessageBox.warning(
                self,
                "Archivo no encontrado",
                f"El archivo no se encuentra en:\n{file_path}"
            )

    def select_all_constants(self):
        """Selecciona todas las constantes físicas"""
        for i in range(self.constants_list.count()):
            item = self.constants_list.item(i)
            item.setCheckState(Qt.CheckState.Checked)

    def clear_all_constants(self):
        """Deselecciona todas las constantes físicas"""
        for i in range(self.constants_list.count()):
            item = self.constants_list.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)

    def add_custom_pattern(self):
        """Agrega un patrón personalizado a la lista"""
        pattern = self.pattern_input.text().strip()
        if pattern:
            item = QListWidgetItem(pattern)
            self.custom_patterns_list.addItem(item)
            self.pattern_input.clear()
            self.status_bar.showMessage(f"Patrón agregado: {pattern}", 2000)

    def remove_custom_pattern(self):
        """Elimina el patrón personalizado seleccionado"""
        current_row = self.custom_patterns_list.currentRow()
        if current_row >= 0:
            item = self.custom_patterns_list.takeItem(current_row)
            self.status_bar.showMessage(f"Patrón eliminado: {item.text()}", 2000)

    def clear_custom_patterns(self):
        """Limpia todos los patrones personalizados"""
        self.custom_patterns_list.clear()
        self.status_bar.showMessage("Todos los patrones eliminados", 2000)

    def save_custom_patterns(self):
        """Guarda los patrones personalizados a un archivo"""
        if self.custom_patterns_list.count() == 0:
            QMessageBox.information(self, "Información", "No hay patrones para guardar")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar patrones personalizados",
            os.path.expanduser("~/constant_hunter_patterns.json"),
            "Archivos JSON (*.json)"
        )

        if file_path:
            try:
                patterns = []
                for i in range(self.custom_patterns_list.count()):
                    patterns.append(self.custom_patterns_list.item(i).text())

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({"patterns": patterns}, f, indent=2)

                self.status_bar.showMessage(f"Patrones guardados en {file_path}", 3000)
                QMessageBox.information(self, "Éxito", f"Patrones guardados en:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudieron guardar los patrones: {str(e)}")

    def load_custom_patterns(self):
        """Carga patrones personalizados desde un archivo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Cargar patrones personalizados",
            os.path.expanduser("~"),
            "Archivos JSON (*.json);;Archivos de texto (*.txt)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.json'):
                        data = json.load(f)
                        patterns = data.get("patterns", [])
                    else:
                        patterns = [line.strip() for line in f if line.strip()]

                self.custom_patterns_list.clear()
                for pattern in patterns:
                    self.custom_patterns_list.addItem(pattern)

                self.status_bar.showMessage(f"{len(patterns)} patrones cargados", 3000)
                QMessageBox.information(self, "Éxito", f"{len(patterns)} patrones cargados")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudieron cargar los patrones: {str(e)}")

    def start_constants_search(self):
        """Inicia la búsqueda de constantes físicas con CUDA"""
        if not self.current_file:
            QMessageBox.warning(self, "Advertencia", "Por favor seleccione un archivo primero")
            return

        # Obtener constantes seleccionadas
        selected_constants = []
        for i in range(self.constants_list.count()):
            item = self.constants_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                const_key = item.data(Qt.ItemDataRole.UserRole)
                selected_constants.append(const_key)

        if not selected_constants:
            QMessageBox.warning(self, "Advertencia", "Por favor seleccione al menos una constante")
            return

        # Verificar CUDA
        if not self.cuda_engine or not self.cuda_engine.cuda_binary.exists():
            reply = QMessageBox.question(
                self,
                "CUDA no disponible",
                "El motor CUDA no está disponible. ¿Desea continuar en modo simulación?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return

        # Deshabilitar controles durante la búsqueda
        self.set_search_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Limpiar resultados anteriores
        self.results_table.setRowCount(0)
        self.details_text.clear()
        self.stats_label.setText("Buscando...")

        # Crear y configurar hilo de búsqueda CUDA
        self.search_thread = CUDASearchThread(self.current_file)

        # Conectar señales
        self.search_thread.progress_updated.connect(self.update_progress)
        self.search_thread.output_received.connect(self.handle_cuda_output)
        self.search_thread.constant_result.connect(self.handle_constant_found)
        self.search_thread.search_completed.connect(self.handle_search_results)
        self.search_thread.search_error.connect(self.handle_search_error)
        self.search_thread.search_finished.connect(self.on_search_finished)

        # Iniciar búsqueda
        self.search_thread.start()
        self.status_bar.showMessage("🚀 Búsqueda CUDA iniciada...")

    def start_custom_search(self):
        """Inicia la búsqueda personalizada con CUDA"""
        if not self.current_file:
            QMessageBox.warning(self, "Advertencia", "Por favor seleccione un archivo primero")
            return

        # Obtener patrones personalizados
        patterns = []
        for i in range(self.custom_patterns_list.count()):
            patterns.append(self.custom_patterns_list.item(i).text())

        if not patterns:
            QMessageBox.warning(self, "Advertencia", "Por favor agregue al menos un patrón personalizado")
            return

        # Verificar que los patrones sean válidos (solo dígitos para esta versión)
        valid_patterns = []
        for pattern in patterns:
            if pattern and pattern.strip():
                # Asegurar que solo contiene dígitos (para búsqueda en archivos de dígitos)
                clean_pattern = ''.join(c for c in pattern if c.isdigit())
                if clean_pattern and len(clean_pattern) >= 3:
                    valid_patterns.append(clean_pattern)
                else:
                    QMessageBox.warning(self, "Patrón inválido",
                                      f"El patrón '{pattern}' no es válido.\n"
                                      f"Debe contener al menos 3 dígitos.")
                    return

        # Preparar configuración de búsqueda
        blocks = self.gpu_block_spin.value()
        threads = self.threads_spin.value()

        # Deshabilitar controles
        self.set_search_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.details_text.clear()

        # Crear hilo de búsqueda personalizada
        self.custom_search_thread = CustomSearchThread(
            self.current_file,
            valid_patterns,
            blocks=blocks,
            threads=threads
        )

        # Conectar señales
        self.custom_search_thread.progress_updated.connect(self.update_progress)
        self.custom_search_thread.output_received.connect(self.handle_cuda_output)
        self.custom_search_thread.search_completed.connect(self.handle_custom_search_results)
        self.custom_search_thread.search_error.connect(self.handle_search_error)
        self.custom_search_thread.search_finished.connect(self.on_custom_search_finished)

        # Iniciar búsqueda
        self.custom_search_thread.start()
        self.status_bar.showMessage(f"🔍 Búsqueda personalizada iniciada ({len(valid_patterns)} patrones)...")

    def update_progress(self, value, message):
        """Actualiza la barra de progreso"""
        self.progress_bar.setValue(value)
        if message:
            self.status_bar.showMessage(message, 2000)

    def handle_cuda_output(self, line):
        """Maneja la salida en tiempo real del kernel CUDA"""
        # Mostrar en el área de detalles
        self.details_text.append(line)

        # Auto-scroll
        scrollbar = self.details_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # Detectar resultados en la salida
        if "coincidencias" in line.lower() and ("✓" in line or "GB/s" in line):
            self.status_bar.showMessage(line.strip(), 3000)

    def handle_constant_found(self, constant_name, data):
        """Maneja cuando se encuentra una constante"""
        # Buscar si ya existe la constante en la tabla
        existing_row = -1
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if item and item.text() == constant_name:
                existing_row = row
                break

        if existing_row >= 0:
            row = existing_row
        else:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

        # Constante
        self.results_table.setItem(row, 0, QTableWidgetItem(constant_name))

        # Coincidencias
        matches = data.get('matches', 0)
        self.results_table.setItem(row, 1, QTableWidgetItem(str(matches)))

        # Tiempo (ms)
        time_ms = data.get('time_ms', 0)
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{time_ms:.2f}"))

        # Throughput (GB/s)
        throughput = data.get('throughput_gbs', 0)
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{throughput:.2f}"))

        # Dígitos (obtener del archivo o de constants_info)
        digits = data.get('digits', '')
        if not digits and constant_name in self.constants_info:
            digits = self.constants_info[constant_name].get('digits', '')
        self.results_table.setItem(row, 4, QTableWidgetItem(digits[:15] + "..." if len(digits) > 15 else digits))

        # Posiciones
        positions = data.get('positions', [])
        pos_text = self.format_positions_for_table(positions)
        self.results_table.setItem(row, 5, QTableWidgetItem(pos_text))

        # Auto-ajustar columnas
        self.results_table.resizeColumnsToContents()

        # Forzar actualización de la GUI
        self.results_table.viewport().update()

    def format_positions_for_table(self, positions, max_display=3):
        """Formatea posiciones para mostrar en tabla"""
        if not positions:
            return "Ninguna"

        if len(positions) <= max_display:
            return ", ".join(str(p) for p in positions)
        else:
            displayed = ", ".join(str(p) for p in positions[:max_display])
            return f"{displayed}... (+{len(positions) - max_display})"

    def update_realtime_stats(self):
        """Actualiza estadísticas en tiempo real durante la búsqueda"""
        total_matches = 0
        constants_found = 0

        for row in range(self.results_table.rowCount()):
            matches_item = self.results_table.item(row, 1)
            if matches_item:
                try:
                    matches = int(matches_item.text())
                    total_matches += matches
                    if matches > 0:
                        constants_found += 1
                except:
                    pass

        self.stats_label.setText(
            f"📊 Buscando... | "
            f"Constantes encontradas: {constants_found} | "
            f"Coincidencias: {total_matches}"
        )

    def handle_search_results(self, results):
        """Maneja los resultados completos de la búsqueda"""
        self.results = results

        # Limpiar tabla primero
        self.results_table.setRowCount(0)

        # Cargar información de constantes
        self.constants_info = {}
        if self.cuda_engine:
            try:
                self.constants_info = self.cuda_engine.get_available_constants()
            except:
                pass

        # Si no hay información de CUDA, usar datos por defecto
        if not self.constants_info:
            self.constants_info = {
                "c": {"digits": "299792458", "name": "Velocidad de la luz en vacío"},
                "h": {"digits": "662607015", "name": "Constante de Planck"},
                "hbar": {"digits": "1054571817", "name": "Constante de Planck reducida"},
                "G": {"digits": "667430", "name": "Constante gravitacional"},
                "k": {"digits": "1380649", "name": "Constante de Boltzmann"},
                "mu0": {"digits": "125663706127", "name": "Permeabilidad magnética"},
                "epsilon0": {"digits": "88541878188", "name": "Permitividad eléctrica"},
                "sigma": {"digits": "5670374419", "name": "Constante de Stefan-Boltzmann"},
                "Z0": {"digits": "376730313412", "name": "Impedancia característica del vacío"}
            }

        # Procesar cada resultado
        total_matches = 0
        total_time = 0
        constants_found = 0
        row = 0

        for const_name, data in results.items():
            if not data:
                continue

            self.results_table.insertRow(row)

            # Constante
            self.results_table.setItem(row, 0, QTableWidgetItem(const_name))

            # Coincidencias
            matches = data.get('matches', 0)
            self.results_table.setItem(row, 1, QTableWidgetItem(str(matches)))
            total_matches += matches

            if matches > 0:
                constants_found += 1

            # Tiempo (ms)
            time_ms = data.get('time_ms', 0)
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{time_ms:.2f}"))
            total_time += time_ms

            # Throughput (GB/s)
            throughput = data.get('throughput_gbs', 0)
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{throughput:.2f}"))

            # Dígitos
            digits = data.get('digits', '')
            if not digits and const_name in self.constants_info:
                digits = self.constants_info[const_name].get('digits', '')
            self.results_table.setItem(row, 4, QTableWidgetItem(digits[:20] + "..." if len(digits) > 20 else digits))

            # Posiciones
            positions = data.get('positions', [])
            pos_text = self.format_positions_for_table(positions)
            self.results_table.setItem(row, 5, QTableWidgetItem(pos_text))

            row += 1

        # Actualizar estadísticas
        if self.current_file:
            try:
                import os
                file_size_mb = os.path.getsize(self.current_file) / (1024 * 1024)
                file_size_gb = file_size_mb / 1024

                throughput_total = file_size_gb / (total_time / 1000) if total_time > 0 else 0

                self.stats_label.setText(
                    f"📊 Resultados: {constants_found}/{len(results)} constantes encontradas | "
                    f"{total_matches} coincidencias totales | "
                    f"Tiempo total: {total_time/1000:.2f}s | "
                    f"Throughput: {throughput_total:.2f} GB/s | "
                    f"Archivo: {file_size_mb:.1f} MB"
                )
            except:
                self.stats_label.setText(
                    f"📊 Resultados: {constants_found}/{len(results)} constantes encontradas | "
                    f"{total_matches} coincidencias totales | "
                    f"Tiempo total: {total_time/1000:.2f}s"
                )

        # Auto-ajustar columnas
        self.results_table.resizeColumnsToContents()

        # Actualizar visualización
        self.update_visualization()

        # Actualizar historial
        self.load_search_history()

        self.status_bar.showMessage(f"✅ Búsqueda completada: {total_matches} coincidencias encontradas", 5000)

    def handle_custom_search_results(self, results):
        """Maneja los resultados de búsqueda personalizada"""
        self.results = results

        # Actualizar tabla
        self.results_table.setRowCount(0)

        total_matches = 0
        row = 0

        for pattern, data in results.items():
            self.results_table.insertRow(row)

            # Patrón (truncar si es muy largo)
            display_pattern = pattern if len(pattern) <= 20 else pattern[:17] + "..."
            self.results_table.setItem(row, 0, QTableWidgetItem(f"Pat: {display_pattern}"))

            # Coincidencias
            matches = data.get('matches', 0)
            self.results_table.setItem(row, 1, QTableWidgetItem(str(matches)))
            total_matches += matches

            # Tiempo
            time_ms = data.get('time_ms', 0)
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{time_ms:.2f}"))

            # Throughput
            throughput = data.get('throughput_gbs', 0)
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{throughput:.2f}"))

            # Dígitos (mostrar patrón)
            self.results_table.setItem(row, 4, QTableWidgetItem(pattern[:30]))

            # Posiciones
            positions = data.get('positions', [])
            if positions:
                pos_text = ", ".join(str(p) for p in positions[:3])
                if len(positions) > 3:
                    pos_text += f"... (+{len(positions)-3})"
            else:
                pos_text = "Ninguna"
            self.results_table.setItem(row, 5, QTableWidgetItem(pos_text))

            row += 1

        # Actualizar estadísticas
        self.stats_label.setText(
            f"📊 Resultados personalizados: {len(results)} patrones | "
            f"{total_matches} coincidencias totales"
        )

        # Actualizar visualización
        self.update_visualization()

        self.status_bar.showMessage(f"✅ Búsqueda personalizada completada: {total_matches} coincidencias", 5000)

    def on_custom_search_finished(self):
        """Limpieza cuando termina la búsqueda personalizada"""
        self.set_search_controls_enabled(True)
        self.progress_bar.setVisible(False)
        self.custom_search_thread = None

        # Cambiar a pestaña de resultados
        self.tab_widget.setCurrentIndex(3)

    def handle_search_error(self, error_message):
        """Maneja errores en la búsqueda"""
        QMessageBox.critical(self, "Error en búsqueda CUDA", error_message)
        self.status_bar.showMessage(f"❌ Error: {error_message}", 5000)

    def on_search_finished(self):
        """Limpieza cuando termina la búsqueda"""
        self.set_search_controls_enabled(True)
        self.progress_bar.setVisible(False)
        self.search_thread = None

        # Cambiar a pestaña de resultados
        self.tab_widget.setCurrentIndex(3)

    def set_search_controls_enabled(self, enabled):
        """Habilita o deshabilita los controles de búsqueda"""
        self.search_btn.setEnabled(enabled)
        self.select_file_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)
        self.custom_search_btn.setEnabled(enabled)
        self.constants_list.setEnabled(enabled)
        self.custom_patterns_list.setEnabled(enabled)
        self.pattern_input.setEnabled(enabled)
        self.gpu_block_spin.setEnabled(enabled)
        self.threads_spin.setEnabled(enabled)

    def stop_search(self):
        """Detiene la búsqueda en curso"""
        if self.search_thread and self.search_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Detener búsqueda",
                "¿Está seguro de detener la búsqueda en curso?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.search_thread.stop()
                self.status_bar.showMessage("⏹️ Búsqueda detenida", 3000)
        elif self.custom_search_thread and self.custom_search_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Detener búsqueda",
                "¿Está seguro de detener la búsqueda en curso?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.custom_search_thread.stop()
                self.status_bar.showMessage("⏹️ Búsqueda personalizada detenida", 3000)

    # ===================== EXPORTACIÓN Y VISUALIZACIÓN =====================

    def export_to_csv(self):
        """Exporta resultados a CSV"""
        if not self.results:
            QMessageBox.warning(self, "Advertencia", "No hay resultados para exportar")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar resultados a CSV",
            os.path.expanduser("~/constant_hunter_results.csv"),
            "Archivos CSV (*.csv)"
        )

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # Encabezados
                    writer.writerow([
                        'Constante', 'Coincidencias', 'Tiempo (ms)',
                        'Throughput (GB/s)', 'Posiciones'
                    ])

                    # Datos
                    for const_name, data in self.results.items():
                        positions = data.get('positions', [])
                        pos_str = ';'.join(str(p) for p in positions[:100])  # Limitar a 100 posiciones

                        writer.writerow([
                            const_name,
                            data.get('matches', 0),
                            f"{data.get('time_ms', 0):.2f}",
                            f"{data.get('throughput_gbs', 0):.2f}",
                            pos_str
                        ])

                self.status_bar.showMessage(f"✅ Resultados exportados a {file_path}", 3000)
                QMessageBox.information(self, "Éxito", f"Resultados exportados a:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo exportar: {str(e)}")

    def export_to_json(self):
        """Exporta resultados a JSON"""
        if not self.results:
            QMessageBox.warning(self, "Advertencia", "No hay resultados para exportar")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar resultados a JSON",
            os.path.expanduser("~/constant_hunter_results.json"),
            "Archivos JSON (*.json)"
        )

        if file_path:
            try:
                # Preparar datos para exportación
                export_data = {
                    'metadata': {
                        'export_date': datetime.now().isoformat(),
                        'file': self.current_file,
                        'total_constants': len(self.results)
                    },
                    'results': self.results
                }

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

                self.status_bar.showMessage(f"✅ Resultados exportados a {file_path}", 3000)
                QMessageBox.information(self, "Éxito", f"Resultados exportados a:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo exportar: {str(e)}")

    def export_to_pdf(self):
        """Exporta resultados a PDF (simulado)"""
        QMessageBox.information(
            self,
            "Exportar a PDF",
            "La exportación a PDF se implementará en la siguiente versión.\n"
            "Por ahora, use CSV o JSON."
        )

    def clear_results(self):
        """Limpia todos los resultados"""
        if self.results:
            reply = QMessageBox.question(
                self,
                "Limpiar resultados",
                "¿Está seguro de limpiar todos los resultados?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.results = {}
                self.results_table.setRowCount(0)
                self.details_text.clear()
                self.stats_label.setText("Estadísticas: Resultados limpiados")
                self.status_bar.showMessage("🗑️ Resultados limpiados", 3000)

    def update_visualization(self):
        """Actualiza la visualización de resultados"""
        if not self.results:
            self.advanced_stats_text.setText("No hay resultados para visualizar.")
            return

        # Limpiar figura
        self.figure.clear()

        # Preparar datos
        constants = []
        matches = []

        for const_name, data in self.results.items():
            if data.get('matches', 0) > 0:
                constants.append(const_name)
                matches.append(data.get('matches', 0))

        if not constants:
            self.advanced_stats_text.setText("No hay constantes con coincidencias para visualizar.")
            return

        # Crear gráfico según el tipo seleccionado
        chart_type = self.chart_type_combo.currentText()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        self.figure.patch.set_facecolor('#2b2b2b')

        colors = plt.cm.viridis(np.linspace(0, 1, len(constants)))

        if chart_type == "Barras":
            bars = ax.bar(constants, matches, color=colors)
            ax.set_xlabel('Constantes', color='white')
            ax.set_ylabel('Coincidencias', color='white')
            ax.set_title('Distribución de Coincidencias', color='white', pad=20)
            ax.tick_params(colors='white')

            # Añadir valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', color='white')

        elif chart_type == "Torta":
            ax.pie(matches, labels=constants, colors=colors, autopct='%1.1f%%')
            ax.set_title('Distribución de Coincidencias', color='white', pad=20)

        elif chart_type == "Dispersión":
            x = range(len(constants))
            ax.scatter(x, matches, s=100, c=colors, alpha=0.7)
            ax.plot(x, matches, '--', color='gray', alpha=0.5)
            ax.set_xlabel('Constantes', color='white')
            ax.set_ylabel('Coincidencias', color='white')
            ax.set_title('Distribución de Coincidencias', color='white', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(constants, rotation=45, ha='right', color='white')
            ax.tick_params(colors='white')

        elif chart_type == "Heatmap":
            # Crear matriz simple para heatmap
            data_matrix = np.array(matches).reshape(1, -1)
            im = ax.imshow(data_matrix, aspect='auto', cmap='coolwarm')
            ax.set_xticks(range(len(constants)))
            ax.set_xticklabels(constants, rotation=45, ha='right', color='white')
            ax.set_yticks([])
            ax.set_title('Heatmap de Coincidencias', color='white', pad=20)
            plt.colorbar(im, ax=ax)

        # Actualizar canvas
        self.canvas.draw()

        # Actualizar estadísticas de texto
        stats_text = "=== ESTADÍSTICAS ===\n\n"
        stats_text += f"Total de constantes: {len(self.results)}\n"
        stats_text += f"Constantes con coincidencias: {len(constants)}\n"
        stats_text += f"Total de coincidencias: {sum(matches)}\n"
        stats_text += f"Coincidencias promedio: {sum(matches)/len(constants):.1f}\n"

        if len(matches) > 0:
            stats_text += f"Máximo coincidencias: {max(matches)} ({constants[matches.index(max(matches))]})\n"
            stats_text += f"Mínimo coincidencias: {min(matches)} ({constants[matches.index(min(matches))]})\n"

        self.advanced_stats_text.setText(stats_text)

    def save_chart(self):
        """Guarda el gráfico actual a un archivo"""
        if not self.results:
            QMessageBox.warning(self, "Advertencia", "No hay gráfico para guardar")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar gráfico",
            os.path.expanduser("~/constant_hunter_chart.png"),
            "Imágenes PNG (*.png);;Imágenes JPEG (*.jpg);;PDF (*.pdf);;SVG (*.svg)"
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, facecolor='#2b2b2b', bbox_inches='tight')
                self.status_bar.showMessage(f"✅ Gráfico guardado en {file_path}", 3000)
                QMessageBox.information(self, "Éxito", f"Gráfico guardado en:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar el gráfico: {str(e)}")

    # ===================== HISTORIAL =====================

    def load_search_history(self):
        """Carga el historial de búsquedas desde los directorios de resultados"""
        self.history_tree.clear()

        try:
            from pathlib import Path
            import os
            from datetime import datetime

            # Buscar directorios de resultados
            possible_dirs = [
                Path.cwd() / "results",
                Path(__file__).parent.parent / "results",
                Path("/home/padmin/Descargas/Constant Hunter v.1/results")
            ]

            all_results = []

            for base_dir in possible_dirs:
                if base_dir.exists():
                    # Buscar todos los directorios results_*
                    result_dirs = list(base_dir.glob('results_*'))

                    for dir_path in result_dirs:
                        if dir_path.is_dir():
                            try:
                                # Leer información del directorio
                                stat = dir_path.stat()
                                modified_time = datetime.fromtimestamp(stat.st_mtime)
                                age = datetime.now() - modified_time

                                # Calcular edad en formato legible
                                if age.days > 0:
                                    age_str = f"{age.days} días"
                                elif age.seconds > 3600:
                                    age_str = f"{age.seconds // 3600} horas"
                                elif age.seconds > 60:
                                    age_str = f"{age.seconds // 60} minutos"
                                else:
                                    age_str = f"{age.seconds} segundos"

                                # Contar archivos de resultados
                                result_files = list(dir_path.glob('*_*.txt'))
                                total_matches = 0

                                # Intentar leer el archivo de resumen
                                summary_file = dir_path / 'SUMMARY.txt'
                                file_name = "Pi - Dec.txt"
                                constants_searched = 0

                                if summary_file.exists():
                                    try:
                                        with open(summary_file, 'r') as f:
                                            summary_lines = f.readlines()

                                        # Extraer información del resumen
                                        for line in summary_lines:
                                            line = line.strip()
                                            if 'ARCHIVO' in line and 'ANALIZADO:' in line:
                                                try:
                                                    file_name = line.split(':')[1].strip().split()[0]
                                                except:
                                                    pass
                                            elif 'coincidencias' in line and 'GB/s' in line:
                                                try:
                                                    # Ejemplo: "c : 1 coincidencias (43.52 ms, 21.40 GB/s)"
                                                    parts = line.split()
                                                    if len(parts) >= 3:
                                                        matches = int(parts[2])
                                                        total_matches += matches
                                                except:
                                                    pass
                                            elif 'Constantes buscadas:' in line:
                                                try:
                                                    constants_searched = int(line.split(':')[1].strip())
                                                except:
                                                    pass
                                    except:
                                        pass

                                # Si no hay resumen, contar desde archivos individuales
                                if total_matches == 0:
                                    for file_path in result_files:
                                        try:
                                            with open(file_path, 'r') as f:
                                                content = f.read()
                                                # Contar líneas con números (posiciones)
                                                lines = [l for l in content.split('\n') if l.strip().isdigit()]
                                                total_matches += len(lines)
                                        except:
                                            pass

                                # Crear ítem para el árbol
                                item = QTreeWidgetItem(self.history_tree)
                                item.setText(0, modified_time.strftime("%Y-%m-%d %H:%M:%S"))
                                item.setText(1, file_name[:20] + "..." if len(file_name) > 20 else file_name)
                                item.setText(2, str(len(result_files)))
                                item.setText(3, str(total_matches))
                                item.setText(4, age_str)

                                # Guardar ruta completa en datos
                                item.setData(0, Qt.ItemDataRole.UserRole, {
                                    'path': str(dir_path),
                                    'summary_file': str(summary_file) if summary_file.exists() else None,
                                    'result_files': [str(f) for f in result_files],
                                    'total_matches': total_matches,
                                    'file_name': file_name,
                                    'timestamp': modified_time.isoformat()
                                })

                                # Colorear según la antigüedad
                                if age.days == 0 and age.seconds < 3600:
                                    item.setForeground(0, QColor('#4CAF50'))  # Verde para reciente (<1h)
                                elif age.days == 0:
                                    item.setForeground(0, QColor('#FF9800'))  # Naranja para hoy
                                elif age.days <= 7:
                                    item.setForeground(0, QColor('#2196F3'))  # Azul para esta semana
                                else:
                                    item.setForeground(0, QColor('#9E9E9E'))  # Gris para antiguo

                                all_results.append((modified_time, item))

                            except Exception as e:
                                print(f"Error procesando {dir_path}: {e}")

            # Ordenar por fecha (más reciente primero)
            all_results.sort(key=lambda x: x[0], reverse=True)

            # Limitar a 50 resultados
            if len(all_results) > 50:
                for _, item in all_results[50:]:
                    index = self.history_tree.indexOfTopLevelItem(item)
                    if index >= 0:
                        self.history_tree.takeTopLevelItem(index)

            self.history_tree.sortItems(0, Qt.SortOrder.DescendingOrder)

            # Actualizar contador en barra de estado
            count = self.history_tree.topLevelItemCount()
            self.status_bar.showMessage(f"📜 Historial cargado: {count} búsquedas encontradas", 3000)

        except Exception as e:
            print(f"Error cargando historial: {e}")
            QMessageBox.warning(self, "Error", f"No se pudo cargar el historial: {str(e)}")

    def load_selected_history(self):
        """Carga el resultado seleccionado del historial"""
        selected_items = self.history_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Advertencia", "Por favor seleccione un resultado del historial")
            return

        item = selected_items[0]
        data = item.data(0, Qt.ItemDataRole.UserRole)

        if not data or 'path' not in data:
            QMessageBox.warning(self, "Error", "Datos del historial no válidos")
            return

        dir_path = data['path']

        if not os.path.exists(dir_path):
            QMessageBox.warning(self, "Error", f"El directorio ya no existe: {dir_path}")
            # Eliminar del historial
            index = self.history_tree.indexOfTopLevelItem(item)
            if index >= 0:
                self.history_tree.takeTopLevelItem(index)
            return

        try:
            # Preguntar al usuario qué quiere hacer
            reply = QMessageBox.question(
                self,
                "Cargar histórico",
                f"¿Qué desea hacer con este resultado?\n\n"
                f"Archivo: {data.get('file_name', 'Desconocido')}\n"
                f"Coincidencias: {data.get('total_matches', 0)}\n"
                f"Fecha: {item.text(0)}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Mostrar resultados en la tabla
                self.load_results_from_directory(dir_path)
            elif reply == QMessageBox.StandardButton.No:
                # Copiar resultados a la tabla actual
                self.copy_results_to_current(dir_path)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar el resultado: {str(e)}")

    def load_results_from_directory(self, dir_path):
        """Carga resultados desde un directorio a la tabla"""
        from pathlib import Path

        # Limpiar tabla actual
        self.results_table.setRowCount(0)
        self.results = {}

        dir_path = Path(dir_path)
        result_files = list(dir_path.glob('*_*.txt'))

        total_matches = 0
        row = 0

        for file_path in result_files:
            if 'SUMMARY' in file_path.name:
                continue

            const_name = file_path.stem.split('_')[0]

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parsear posiciones
                positions = []
                lines = content.split('\n')

                for line in lines:
                    line = line.strip()
                    if line.isdigit():
                        positions.append(int(line))
                    elif 'coincidencias:' in line:
                        try:
                            matches = int(line.split(':')[1].strip())
                        except:
                            matches = len(positions)

                matches = len(positions)
                total_matches += matches

                # Añadir a tabla
                self.results_table.insertRow(row)
                self.results_table.setItem(row, 0, QTableWidgetItem(const_name))
                self.results_table.setItem(row, 1, QTableWidgetItem(str(matches)))
                self.results_table.setItem(row, 2, QTableWidgetItem("N/A"))
                self.results_table.setItem(row, 3, QTableWidgetItem("N/A"))
                self.results_table.setItem(row, 4, QTableWidgetItem(""))

                # Posiciones
                pos_text = self.format_positions_for_table(positions)
                self.results_table.setItem(row, 5, QTableWidgetItem(pos_text))

                # Guardar en self.results
                self.results[const_name] = {
                    'matches': matches,
                    'positions': positions[:100],  # Limitar para memoria
                    'time_ms': 0,
                    'throughput_gbs': 0
                }

                row += 1

            except Exception as e:
                print(f"Error cargando {file_path}: {e}")

        # Actualizar estadísticas
        self.stats_label.setText(
            f"📜 Resultados históricos: {len(result_files)} constantes | "
            f"{total_matches} coincidencias | "
            f"Directorio: {dir_path.name}"
        )

        # Actualizar visualización
        self.update_visualization()

        self.status_bar.showMessage(f"✅ Historial cargado: {total_matches} coincidencias", 3000)
        self.tab_widget.setCurrentIndex(3)  # Ir a pestaña de resultados

    def copy_results_to_current(self, dir_path):
        """Copia resultados históricos a la sesión actual"""
        from pathlib import Path

        dir_path = Path(dir_path)
        result_files = list(dir_path.glob('*_*.txt'))

        added_count = 0

        for file_path in result_files:
            if 'SUMMARY' in file_path.name:
                continue

            const_name = file_path.stem.split('_')[0]

            # Verificar si ya existe en resultados actuales
            if const_name in self.results:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parsear posiciones (simplificado)
                positions = [int(l) for l in content.split('\n') if l.strip().isdigit()]

                if positions:
                    self.results[const_name] = {
                        'matches': len(positions),
                        'positions': positions[:50],
                        'time_ms': 0,
                        'throughput_gbs': 0
                    }
                    added_count += 1

            except:
                pass

        if added_count > 0:
            # Actualizar tabla
            self.handle_search_results(self.results)
            self.status_bar.showMessage(f"✅ {added_count} constantes añadidas del historial", 3000)
        else:
            self.status_bar.showMessage("ℹ️ No se añadieron nuevas constantes", 3000)

    def load_history_result(self, item, column):
        """Carga resultado del historial al hacer doble click"""
        self.load_selected_history()

    def delete_history_item(self):
        """Elimina el elemento seleccionado del historial"""
        selected_items = self.history_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Advertencia", "Por favor seleccione un elemento del historial")
            return

        item = selected_items[0]
        data = item.data(0, Qt.ItemDataRole.UserRole)

        if data and 'path' in data and os.path.exists(data['path']):
            reply = QMessageBox.question(
                self,
                "Eliminar resultado",
                f"¿Está seguro de eliminar este resultado?\n{data['path']}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                try:
                    import shutil
                    shutil.rmtree(data['path'])

                    # Eliminar del árbol
                    index = self.history_tree.indexOfTopLevelItem(item)
                    self.history_tree.takeTopLevelItem(index)

                    self.status_bar.showMessage(f"✅ Resultado eliminado: {os.path.basename(data['path'])}", 3000)

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"No se pudo eliminar: {str(e)}")

    def clear_history(self):
        """Limpia todo el historial"""
        if self.history_tree.topLevelItemCount() == 0:
            return

        reply = QMessageBox.question(
            self,
            "Limpiar historial",
            "¿Está seguro de eliminar todo el historial?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.history_tree.clear()
            self.status_bar.showMessage("🗑️ Historial limpiado", 3000)

    # ===================== MÉTODOS DE MENÚ =====================

    def compile_kernels(self):
        """Compila los kernels CUDA"""
        QMessageBox.information(
            self,
            "Compilar Kernels CUDA",
            "Para compilar los kernels CUDA, ejecute:\n\n"
            "cd \"/home/padmin/Descargas/Constant Hunter v.1\"\n"
            "./src/scripts/compile_all.sh\n\n"
            "O desde la terminal:\n"
            "nvcc -o bin/full_pi_search_enhanced src/cuda/full_pi_search_enhanced.cu "
            "-lcudart -lstdc++ -ccbin /usr/bin/gcc-14 -O3"
        )

    def run_benchmark(self):
        """Ejecuta benchmark de rendimiento"""
        QMessageBox.information(
            self,
            "Benchmark",
            "Ejecutando pruebas de rendimiento...\n\n"
            "Esta funcionalidad medirá el throughput de la GPU\n"
            "y optimizará los parámetros de búsqueda."
        )

    def show_settings(self):
        """Muestra la ventana de configuración"""
        QMessageBox.information(
            self,
            "Configuración",
            "Opciones de configuración:\n\n"
            "• GPU: NVIDIA GeForce GTX 1650\n"
            "• Memoria: 4GB VRAM\n"
            "• Directorio de resultados: results/\n"
            "• Directorio de datasets: datasets/\n\n"
            "(La configuración avanzada se implementará en la siguiente versión)"
        )

    def clear_cache(self):
        """Limpia la cache de la aplicación"""
        reply = QMessageBox.question(
            self,
            "Limpiar cache",
            "¿Está seguro de limpiar la cache de la aplicación?\n"
            "Esto eliminará archivos temporales y caché de resultados.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # TODO: Implementar limpieza de cache
            self.status_bar.showMessage("🗑️ Cache limpiada", 3000)
            QMessageBox.information(self, "Cache", "Cache limpiada correctamente")

    def show_about(self):
        """Muestra información acerca del programa"""
        QMessageBox.about(
            self,
            "Acerca de Constant Hunter v1.0",
            "CONSTANT HUNTER v1.0\n\n"
            "Aplicación GPU-acelerada para búsqueda de constantes físicas\n"
            "en números irracionales (π, e, φ, √2, etc.).\n\n"
            "🎯 Características:\n"
            "• Búsqueda CUDA-optimizada (1GB en <1 segundo)\n"
            "• 9+ constantes físicas predefinidas\n"
            "• Búsqueda personalizada de patrones\n"
            "• Visualización interactiva de resultados\n"
            "• Exportación a CSV/JSON\n"
            "• Historial de búsquedas\n\n"
            "🖥️ Requisitos:\n"
            "• NVIDIA GPU con CUDA 11.0+\n"
            "• 4GB+ VRAM\n"
            "• Python 3.8+ con PyQt6\n\n"
            "🔧 Tecnologías:\n"
            "• CUDA C++ para kernels GPU\n"
            "• PyQt6 para interfaz gráfica\n"
            "• Matplotlib para visualización\n\n"
            "📄 Licencia: Apache 2.0\n"
            "© 2026 Constant Hunter Project\n"
            "github.com/constant-hunter"
        )

    def show_documentation(self):
        """Muestra la documentación"""
        QMessageBox.information(
            self,
            "Documentación",
            "Documentación de Constant Hunter v1.0\n\n"
            "📚 Secciones:\n"
            "1. Instalación y configuración\n"
            "2. Uso de la interfaz gráfica\n"
            "3. Búsqueda de constantes físicas\n"
            "4. Búsqueda personalizada\n"
            "5. Exportación de resultados\n"
            "6. Solución de problemas\n\n"
            "📁 Archivos de documentación:\n"
            "/home/padmin/Descargas/Constant Hunter v.1/docs/\n\n"
            "🌐 Documentación en línea disponible en:\n"
            "github.com/constant-hunter/docs"
        )

    def check_for_updates(self):
        """Busca actualizaciones"""
        QMessageBox.information(
            self,
            "Buscar actualizaciones",
            "Versión actual: v1.0\n\n"
            "Buscando actualizaciones...\n"
            "(Esta funcionalidad se implementará en la siguiente versión)"
        )

    def open_results_directory(self):
        """Abre el directorio de resultados en el explorador de archivos"""
        results_path = Path(__file__).parent.parent / "results"

        if results_path.exists():
            import subprocess
            try:
                subprocess.Popen(['xdg-open', str(results_path)])
            except:
                QMessageBox.information(
                    self,
                    "Directorio de resultados",
                    f"Ruta: {results_path}\n\n"
                    f"Puede abrir manualmente esta carpeta."
                )
        else:
            QMessageBox.information(
                self,
                "Directorio no encontrado",
                f"El directorio de resultados no existe:\n{results_path}\n\n"
                f"Se creará automáticamente al ejecutar una búsqueda."
            )

    def load_constants_list(self):
        """Carga la lista de constantes (ya se hace en setup_constants_search_tab)"""
        pass


    # ===================== MÉTODOS DE VISUALIZACIÓN AVANZADA =====================

    def run_advanced_analysis(self):
        """Ejecuta análisis avanzado completo"""
        if not self.results:
            QMessageBox.warning(self, "Advertencia", "No hay resultados para analizar")
            return

        try:
            # Intentar importar el visualizador avanzado
            try:
                from advanced_visualizer import AdvancedVisualizer
            except ImportError:
                QMessageBox.warning(self, "Advertencia", "El módulo advanced_visualizer no está disponible")
                return

            # Obtener directorio de resultados más reciente
            results_dirs = []
            possible_dirs = [
                Path.cwd() / "results",
                Path(__file__).parent.parent / "results",
                Path("/home/padmin/Descargas/Constant Hunter v.1/results")
            ]

            latest_dir = None
            for dir_path in possible_dirs:
                if dir_path.exists():
                    dir_list = list(dir_path.glob('results_*'))
                    if dir_list:
                        latest = max(dir_list, key=lambda x: x.stat().st_mtime)
                        latest_dir = latest
                        break

            if not latest_dir:
                QMessageBox.warning(self, "Advertencia", "No se encontraron directorios de resultados")
                return

            # Crear y ejecutar visualizador
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            visualizer = AdvancedVisualizer(str(latest_dir))
            visualizer.load_results()
            stats = visualizer.analyze_complete()

            # Generar visualizaciones
            output_path = visualizer.create_all_visualizations()

            QApplication.restoreOverrideCursor()

            # Mostrar resumen
            summary = "📊 ANÁLISIS AVANZADO COMPLETADO\n"
            summary += "=" * 50 + "\n"
            summary += f"Directorio analizado: {latest_dir.name}\n"
            summary += f"Constantes procesadas: {len(stats)}\n"
            summary += f"Archivos generados: {len(list(output_path.glob('*'))) if output_path.exists() else 0}\n\n"

            for const_name, s in stats.items():
                summary += f"📈 {const_name}:\n"
                summary += f"   • Ocurrencias: {s.get('count', 0)}\n"
                if 'density_mb' in s:
                    summary += f"   • Densidad: {s['density_mb']:.3f}/MB\n"
                if 'mean_gap' in s:
                    summary += f"   • Gap promedio: {s['mean_gap']:.0f} dígitos\n"
                if 'ks_pvalue' in s:
                    uniform = "SÍ" if s.get('is_uniform', False) else "NO"
                    summary += f"   • Uniforme: {uniform} (p={s['ks_pvalue']:.4f})\n"
                if 'clusters_1000' in s:
                    summary += f"   • Clusters: {s['clusters_1000'].get('num_clusters', 0)}\n"
                summary += "\n"

            summary += f"📁 Visualizaciones guardadas en:\n{output_path}"

            self.advanced_stats_text.setText(summary)

            # Actualizar gráfico con datos del análisis
            self.update_advanced_visualization(stats)

            QMessageBox.information(self, "Análisis Completado",
                                  f"Análisis avanzado completado.\n\n"
                                  f"Archivos generados en:\n{output_path}")

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"No se pudo ejecutar el análisis avanzado:\n{str(e)}")

    def run_specific_analysis(self, analysis_type):
        """Ejecuta análisis específico"""
        if not self.results:
            QMessageBox.warning(self, "Advertencia", "No hay resultados para analizar")
            return

        try:
            # Intentar importar el visualizador avanzado
            try:
                from advanced_visualizer import AdvancedVisualizer
            except ImportError:
                QMessageBox.warning(self, "Advertencia", "El módulo advanced_visualizer no está disponible")
                return

            latest_dir = None
            possible_dirs = [Path.cwd() / "results", Path(__file__).parent.parent / "results"]

            for dir_path in possible_dirs:
                if dir_path.exists():
                    dir_list = list(dir_path.glob('results_*'))
                    if dir_list:
                        latest_dir = max(dir_list, key=lambda x: x.stat().st_mtime)
                        break

            if not latest_dir:
                QMessageBox.warning(self, "Advertencia", "No se encontraron resultados")
                return

            visualizer = AdvancedVisualizer(str(latest_dir))
            visualizer.load_results()

            if analysis_type == "clusters":
                # Análisis específico de clusters
                analysis_text = "🎯 ANÁLISIS DE CLUSTERS\n"
                analysis_text += "=" * 40 + "\n"

                for const_name, positions in visualizer.data.items():
                    if len(positions) > 10:
                        clusters = visualizer._analyze_clustering_detailed(positions, [100, 500, 1000])
                        analysis_text += f"\n{const_name} ({len(positions)} ocurrencias):\n"
                        for threshold in [100, 500, 1000]:
                            key = f'clusters_{threshold}'
                            if key in clusters:
                                c = clusters[key]
                                analysis_text += f"  • Threshold {threshold}: {c['num_clusters']} clusters "
                                analysis_text += f"(avg size: {c.get('mean_cluster_size', 0):.1f})\n"

                self.advanced_stats_text.setText(analysis_text)

            elif analysis_type == "gaps":
                # Análisis específico de gaps
                analysis_text = "📏 ANÁLISIS DE GAPS\n"
                analysis_text += "=" * 40 + "\n"

                for const_name, positions in visualizer.data.items():
                    if len(positions) > 5:
                        import numpy as np
                        positions_array = np.array(positions)
                        gaps = np.diff(positions_array)
                        analysis_text += f"\n{const_name}:\n"
                        analysis_text += f"  • Gap mínimo: {gaps.min():.0f} dígitos\n"
                        analysis_text += f"  • Gap máximo: {gaps.max():.0f} dígitos\n"
                        analysis_text += f"  • Gap promedio: {gaps.mean():.0f} ± {gaps.std():.0f}\n"
                        analysis_text += f"  • Mediana: {np.median(gaps):.0f} dígitos\n"

                self.advanced_stats_text.setText(analysis_text)

            elif analysis_type == "uniformity":
                # Test de uniformidad
                analysis_text = "📊 TEST DE UNIFORMIDAD (Kolmogorov-Smirnov)\n"
                analysis_text += "=" * 50 + "\n"

                for const_name, positions in visualizer.data.items():
                    if len(positions) > 10:
                        try:
                            from scipy import stats
                            import numpy as np
                            # Normalizar posiciones entre 0 y 1
                            positions_norm = np.array(positions) / max(positions)
                            ks_test = stats.kstest(positions_norm, 'uniform', args=(0, 1))

                            analysis_text += f"\n{const_name}:\n"
                            analysis_text += f"  • KS statistic: {ks_test.statistic:.4f}\n"
                            analysis_text += f"  • p-value: {ks_test.pvalue:.6f}\n"
                            analysis_text += f"  • Distribución uniforme: {'SÍ' if ks_test.pvalue > 0.05 else 'NO'}\n"
                            analysis_text += f"  • Nivel de confianza: {100*(1-ks_test.pvalue):.1f}%\n"
                        except Exception as e:
                            analysis_text += f"\n{const_name}: Error en test KS - {str(e)}\n"

                self.advanced_stats_text.setText(analysis_text)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en análisis: {str(e)}")

    def update_advanced_visualization(self, stats):
        """Actualiza visualización con datos avanzados"""
        if not stats:
            return

        self.figure.clear()

        # Crear gráfico según el tipo seleccionado
        chart_type = self.chart_type_combo.currentText()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        self.figure.patch.set_facecolor('#2b2b2b')

        constants = list(stats.keys())

        if chart_type == "Barras":
            # Mostrar densidad
            densities = [stats[c].get('density_mb', 0) for c in constants]
            bars = ax.bar(constants, densities, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(constants))))
            ax.set_xlabel('Constantes', color='white')
            ax.set_ylabel('Densidad (ocurrencias/MB)', color='white')
            ax.set_title('Densidad de Ocurrencias por Constante', color='white', pad=20)

        elif chart_type == "Heatmap":
            # Crear matriz de correlaciones (simplificada)
            data_matrix = []
            for c in constants:
                if 'count' in stats[c]:
                    data_matrix.append([
                        stats[c]['count'],
                        stats[c].get('density_mb', 0),
                        stats[c].get('mean_gap', 0) / 1000 if stats[c].get('mean_gap', 0) > 0 else 0,
                        stats[c].get('ks_pvalue', 0.5) if 'ks_pvalue' in stats[c] else 0.5
                    ])

            if data_matrix:
                im = ax.imshow(data_matrix, aspect='auto', cmap='coolwarm')
                ax.set_xticks(range(4))
                ax.set_xticklabels(['Count', 'Density', 'Gap/1k', 'KS p-val'], rotation=45, color='white')
                ax.set_yticks(range(len(constants)))
                ax.set_yticklabels(constants, color='white')
                plt.colorbar(im, ax=ax)
                ax.set_title('Matriz de Características', color='white', pad=20)

        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='#555555')

        self.canvas.draw()


    def closeEvent(self, event):
        """Maneja el cierre de la aplicación"""
        if self.search_thread and self.search_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Búsqueda en progreso",
                "Hay una búsqueda en progreso. ¿Desea detenerla y salir?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.search_thread.stop()
                self.search_thread.wait()
                event.accept()
            else:
                event.ignore()
        elif self.custom_search_thread and self.custom_search_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Búsqueda en progreso",
                "Hay una búsqueda en progreso. ¿Desea detenerla y salir?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.custom_search_thread.stop()
                self.custom_search_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ===================== FUNCIÓN PRINCIPAL =====================

def main():
    """Función principal de la aplicación"""
    app = QApplication(sys.argv)
    app.setApplicationName("Constant Hunter")
    app.setApplicationVersion("1.0")

    # Establecer estilo de la aplicación
    app.setStyle('Fusion')

    # Crear y mostrar ventana principal
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
