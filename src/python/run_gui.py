#!/usr/bin/env python3
"""
Script de lanzamiento mejorado para Constant Hunter GUI
"""

import sys
import os
import traceback

# Añadir el directorio actual al path
sys.path.insert(0, os.path.dirname(__file__))

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    missing_deps = []
    
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        missing_deps.append("PyQt6")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    # Nota: La verificación del binario CUDA se hace en cuda_wrapper.py
    # donde se buscan múltiples ubicaciones posibles
    
    return missing_deps

def check_cuda_engine():
    """Verifica que el motor CUDA puede inicializarse"""
    try:
        from cuda_wrapper import CUDASearchEngine
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        engine = CUDASearchEngine(project_root)
        
        if not engine.cuda_binary.exists():
            print(f"⚠️  Binario CUDA no encontrado: {engine.cuda_binary}")
            print("   Ejecute: ./src/scripts/compile_all.sh")
            return False
        
        # Verificar archivos de datos
        datasets_dir = engine.project_root / "datasets"
        if not datasets_dir.exists():
            print(f"⚠️  Directorio de datasets no encontrado: {datasets_dir}")
            print("   Copie archivos a: datasets/")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error inicializando motor CUDA: {e}")
        return False

def main():
    """Función principal"""
    print("=" * 60)
    print("   CONSTANT HUNTER v1.0 - Búsqueda GPU-Acelerada")
    print("=" * 60)
    print()
    
    # Verificar dependencias
    missing = check_dependencies()
    if missing:
        print(f"❌ Faltan dependencias: {', '.join(missing)}")
        print("\nInstale con:")
        print("  pip install PyQt6 matplotlib numpy pandas")
        sys.exit(1)
    
    # Verificar motor CUDA (solo mostrar advertencia, no salir)
    if not check_cuda_engine():
        print("\n⚠️  Problemas con el motor CUDA detectados")
        print("   La GUI funcionará pero las búsquedas pueden fallar.")
        print("   Presione Enter para continuar o Ctrl+C para salir.")
        try:
            input()
        except KeyboardInterrupt:
            print("\n👋 Saliendo...")
            sys.exit(0)
    
    try:
        from gui_app import main as gui_main
        print("✅ Interfaz gráfica cargada correctamente")
        print("✅ Motor CUDA verificado")
        print("\n🚀 Iniciando aplicación...")
        print()
        
        gui_main()
        
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        print("\nTraceback completo:")
        traceback.print_exc()
        
        input("\nPresione Enter para salir...")
        sys.exit(1)

if __name__ == '__main__':
    main()
