#!/bin/bash
# Script: setup_constant_hunter.sh
# Descripción: Crea la estructura completa del proyecto Constant Hunter v.1

echo "============================================"
echo "  CONFIGURANDO CONSTANT HUNTER v.1"
echo "============================================"

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================
PROJECT_DIR="/home/padmin/Descargas/Constant Hunter v.1"
PROTOTYPE_DIR="/home/padmin/Descargas/Pi/Kernel CUDA"

echo "📁 Directorio del proyecto: $PROJECT_DIR"
echo "📁 Directorio prototipo: $PROTOTYPE_DIR"
echo ""

# ============================================================================
# 1. CREAR ESTRUCTURA DE DIRECTORIOS
# ============================================================================
echo "1️⃣ Creando estructura de directorios..."

mkdir -p "$PROJECT_DIR/src/cuda"               # Kernels CUDA
mkdir -p "$PROJECT_DIR/src/python"             # Código Python/GUI
mkdir -p "$PROJECT_DIR/src/scripts"            # Scripts de utilidad
mkdir -p "$PROJECT_DIR/datasets"               # Archivos de números irracionales
mkdir -p "$PROJECT_DIR/results"                # Resultados de búsqueda
mkdir -p "$PROJECT_DIR/docs"                   # Documentación
mkdir -p "$PROJECT_DIR/bin"                    # Ejecutables compilados
mkdir -p "$PROJECT_DIR/tests"                  # Pruebas unitarias

echo "✅ Estructura de directorios creada"
echo ""

# ============================================================================
# 2. COPIAR ARCHIVOS DEL PROTOTIPO
# ============================================================================
echo "2️⃣ Copiando archivos del prototipo..."

# 2.1 Copiar archivos CUDA (solo las versiones más avanzadas)
echo "📄 Copiando kernels CUDA..."
cp "$PROTOTYPE_DIR/src/full_pi_search_enhanced.cu" "$PROJECT_DIR/src/cuda/"
cp "$PROTOTYPE_DIR/src/search_kernel.cu" "$PROJECT_DIR/src/cuda/"

# 2.2 Copiar TODA la base de datos
echo "📊 Copiando base de datos completa..."
cp -r "$PROTOTYPE_DIR/Database/"* "$PROJECT_DIR/datasets/"

# 2.3 Copiar scripts existentes
echo "⚙️ Copiando scripts..."
cp "$PROTOTYPE_DIR/compile_and_run.sh" "$PROJECT_DIR/src/scripts/"
cp "$PROTOTYPE_DIR/compile_enhanced.sh" "$PROJECT_DIR/src/scripts/"

# 2.4 Copiar resultados de ejemplo (opcional)
echo "📋 Copiando ejemplos de resultados..."
mkdir -p "$PROJECT_DIR/docs/examples"
find "$PROTOTYPE_DIR/results" -name "*.txt" -exec cp {} "$PROJECT_DIR/docs/examples/" \; 2>/dev/null || true

echo "✅ Archivos copiados correctamente"
echo ""

# ============================================================================
# 3. CREAR ARCHIVOS DE CONFIGURACIÓN
# ============================================================================
echo "3️⃣ Creando archivos de configuración..."

# 3.1 requirements.txt para Python
cat > "$PROJECT_DIR/requirements.txt" << 'EOF'
# Constant Hunter v.1 - Dependencias Python
PyQt6==6.6.0
matplotlib==3.8.0
numpy==1.26.0
scipy==1.11.0
seaborn==0.13.0
pandas==2.1.0
pycuda==2023.1
colorama==0.4.6
tqdm==4.66.0
EOF

# 3.2 CMakeLists.txt para compilación CUDA
cat > "$PROJECT_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(ConstantHunter)

set(CMAKE_CXX_STANDARD 14)

# Buscar CUDA
find_package(CUDA REQUIRED)

# Configurar flags de compilación
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -Xptxas -O3 --generate-line-info")
set(CUDA_HOST_COMPILER /usr/bin/gcc-14)

# Ejecutable principal
cuda_add_executable(constant_hunter_cuda
    src/cuda/full_pi_search_enhanced.cu
)

# Instalación
install(TARGETS constant_hunter_cuda DESTINATION bin)
EOF

# 3.3 Script de compilación mejorado
cat > "$PROJECT_DIR/src/scripts/compile_all.sh" << 'EOF'
#!/bin/bash
# Script de compilación para Constant Hunter v.1

echo "========================================"
echo "  COMPILANDO CONSTANT HUNTER v.1"
echo "========================================"

PROJECT_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
echo "Directorio del proyecto: $PROJECT_ROOT"

# Crear directorio bin si no existe
mkdir -p "$PROJECT_ROOT/bin"

# Compilar versión optimizada
echo ""
echo "🛠️  Compilando full_pi_search_enhanced.cu..."
cd "$PROJECT_ROOT"
nvcc -o bin/full_pi_search_enhanced \
    src/cuda/full_pi_search_enhanced.cu \
    -lcudart -lstdc++ \
    -ccbin /usr/bin/gcc-14 \
    -O3 -Xptxas -O3

if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa: bin/full_pi_search_enhanced"
    chmod +x bin/full_pi_search_enhanced
else
    echo "❌ Error en la compilación"
    exit 1
fi

# Compilar kernel básico
echo ""
echo "🛠️  Compilando search_kernel.cu..."
nvcc -o bin/search_kernel_test \
    src/cuda/search_kernel.cu \
    -lcudart -lstdc++ \
    -ccbin /usr/bin/gcc-14 \
    -O3 -Xptxas -O3

if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa: bin/search_kernel_test"
    chmod +x bin/search_kernel_test
fi

echo ""
echo "========================================"
echo "  COMPILACIÓN COMPLETADA"
echo "========================================"
echo ""
echo "Ejecutables disponibles en:"
echo "  $PROJECT_ROOT/bin/"
echo ""
echo "Para ejecutar:"
echo "  ./bin/full_pi_search_enhanced"
EOF

chmod +x "$PROJECT_DIR/src/scripts/compile_all.sh"

# 3.4 README.md inicial
cat > "$PROJECT_DIR/README.md" << 'EOF'
# Constant Hunter v.1

## 🎯 Descripción
Aplicación GPU-acelerada para buscar constantes físicas y secuencias numéricas en números irracionales (π, √2, φ, e, etc.)

## 🚀 Características
- **Búsqueda GPU-acelerada**: 1GB en <1 segundo
- **Interfaz gráfica PyQt6**: Fácil de usar
- **Múltiples constantes físicas**: 9 constantes predefinidas
- **Búsqueda personalizada**: Patrones arbitrarios
- **Exportación avanzada**: CSV, JSON, reportes PDF

## 📁 Estructura del Proyecto

Constant Hunter v.1/
├── src/
│ ├── cuda/ # Kernels CUDA optimizados
│ ├── python/ # Interfaz gráfica PyQt6
│ └── scripts/ # Scripts de utilidad
├── datasets/ # Archivos de números irracionales
├── results/ # Resultados de búsqueda
├── docs/ # Documentación
├── bin/ # Ejecutables compilados
└── tests/ # Pruebas unitarias
## ⚡ Instalación Rápida

### 1. Dependencias del sistema (Fedora 43)
```bash
sudo dnf install gcc14 gcc14-c++ cuda-toolkit
pip install -r requirements.txt

./src/scripts/compile_all.sh

./bin/full_pi_search_enhanced

🔧 Constantes Físicas Predefinidas

    c: Velocidad de la luz en vacío

    h: Constante de Planck

    G: Constante gravitacional

    k: Constante de Boltzmann

    y más...

📊 Resultados Esperados

    Throughput: 20-25 GB/s

    Tiempo de búsqueda: <1 segundo por GB

    Precisión: 100% para búsqueda exacta

📄 Licencia

MIT License
EOF

echo "✅ Archivos de configuración creados"
echo ""
============================================================================
4. VERIFICAR ESTRUCTURA CREADA
============================================================================

echo "4️⃣ Verificando estructura del proyecto..."

echo "📋 Contenido de $PROJECT_DIR:"
find "$PROJECT_DIR" -type d | sort | sed 's|./||' | while read dir; do
depth=$(echo "$dir" | tr -cd '/' | wc -c)
indent=$(printf "%s" $((depth*2)) "")
echo "${indent}📁 $dir"
done

echo ""
echo "📊 Archivos copiados:"
echo " • CUDA: $(find "$PROJECT_DIR/src/cuda" -name ".cu" | wc -l) archivos .cu"
echo " • Datos: $(find "$PROJECT_DIR/datasets" -name ".txt" | wc -l) archivos .txt"
echo " • Scripts: $(find "$PROJECT_DIR/src/scripts" -name "*.sh" | wc -l) archivos .sh"
echo ""
============================================================================
5. CONFIGURAR PERMISOS
============================================================================

echo "5️⃣ Configurando permisos..."

chmod +x "$PROJECT_DIR/src/scripts/"*.sh
echo "✅ Permisos configurados"
============================================================================
6. PRUEBA RÁPIDA DE ARCHIVOS
============================================================================

echo ""
echo "6️⃣ Realizando prueba rápida..."

if [ -f "$PROJECT_DIR/datasets/Pi - Dec.txt" ]; then
PI_SIZE=$(du -h "$PROJECT_DIR/datasets/Pi - Dec.txt" | cut -f1)
echo "✅ Archivo Pi encontrado: $PI_SIZE"
else
echo "⚠️ Advertencia: No se encontró el archivo Pi - Dec.txt"
fi

if [ -f "$PROJECT_DIR/src/cuda/full_pi_search_enhanced.cu" ]; then
echo "✅ Kernel CUDA principal encontrado"
else
echo "❌ Error: No se encontró el kernel CUDA"
fi

echo ""
echo "============================================"
echo " ✅ CONFIGURACIÓN COMPLETADA"
echo "============================================"
echo ""
echo "📌 Siguientes pasos:"
echo ""
echo "1. Compilar los kernels CUDA:"
echo " cd "$PROJECT_DIR""
echo " ./src/scripts/compile_all.sh"
echo ""
echo "2. Instalar dependencias Python:"
echo " pip install -r requirements.txt"
echo ""
echo "3. Para la FASE 1 (GUI PyQt6), ejecutar:"
echo " cd "$PROJECT_DIR/src/python""
echo " # Aquí crearemos gui_app.py en el siguiente paso"
echo ""
echo "4. Verificar que todo funciona:"
echo " ./bin/full_pi_search_enhanced"
echo ""
echo "🎯 Listo para comenzar el desarrollo de la interfaz gráfica!"


## 📝 Instrucciones de Ejecución:

### Paso 1: Guardar el script
```bash
# Crear el archivo de configuración
cat > setup_constant_hunter.sh << 'EOF'
[PEGA EL CONTENIDO COMPLETO DEL SCRIPT ARRIBA AQUÍ]
EOF

# Hacerlo ejecutable
chmod +x setup_constant_hunter.sh

