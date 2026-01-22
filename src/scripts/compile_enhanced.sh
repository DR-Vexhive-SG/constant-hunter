#!/bin/bash

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Función para imprimir con colores
print_color() {
    echo -e "${2}${1}${NC}"
}

# Función para verificar comandos
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_color "✗ Error: $1 no encontrado" "$RED"
        return 1
    fi
    return 0
}

# Función para limpiar archivos temporales
cleanup() {
    if [ -f "nvcc.log" ]; then
        rm nvcc.log
    fi
    if [ -f "ptxas.log" ]; then
        rm ptxas.log
    fi
}

# Capturar señales para limpieza
trap cleanup EXIT INT TERM

print_color "=== Optimizador Avanzado CUDA para Constant Hunter v1.0 ===" "$MAGENTA"
print_color "Fecha: $(date)" "$CYAN"
echo ""

# Determinar directorio del proyecto de forma robusta
PROJECT_ROOT=""
if [ -n "$1" ] && [ "$1" != "--debug" ] && [ "$1" != "--benchmark" ]; then
    PROJECT_ROOT="$1"
    shift
else
    # Buscar hacia arriba desde el script o directorio actual
    SCRIPT_DIR="$(dirname "$(realpath "$0")")"

    # Lista de ubicaciones posibles
    possible_roots=(
        "$(pwd)"
        "$SCRIPT_DIR/.."
        "$SCRIPT_DIR/../.."
        "/home/padmin/Descargas/Constant Hunter v.1"
        "$HOME/Descargas/Constant Hunter v.1"
    )

    for root in "${possible_roots[@]}"; do
        if [ -f "$root/src/cuda/full_pi_search_enhanced.cu" ]; then
            PROJECT_ROOT="$root"
            print_color "✓ Proyecto encontrado en: $PROJECT_ROOT" "$GREEN"
            break
        fi
    done
fi

if [ -z "$PROJECT_ROOT" ] || [ ! -f "$PROJECT_ROOT/src/cuda/full_pi_search_enhanced.cu" ]; then
    print_color "✗ Error: No se pudo encontrar el proyecto Constant Hunter" "$RED"
    echo ""
    print_color "Busqué en:" "$YELLOW"
    for root in "${possible_roots[@]:-$(pwd)}"; do
        echo "  $root"
    done
    echo ""
    print_color "Uso: $0 [ruta_del_proyecto] [--debug] [--benchmark]" "$YELLOW"
    echo "Ej: $0 \"/home/padmin/Descargas/Constant Hunter v.1\" --benchmark"
    exit 1
fi

cd "$PROJECT_ROOT"

# Procesar flags adicionales
DEBUG_MODE=0
BENCHMARK_MODE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=1
            shift
            ;;
        --benchmark)
            BENCHMARK_MODE=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Verificar estructura del proyecto
print_color "📁 Verificando estructura del proyecto..." "$CYAN"
if [ ! -d "src/cuda" ]; then
    print_color "✗ Error: No se encuentra src/cuda/" "$RED"
    exit 1
fi

if [ ! -f "src/cuda/full_pi_search_enhanced.cu" ]; then
    print_color "✗ Error: No se encuentra full_pi_search_enhanced.cu" "$RED"
    exit 1
fi

print_color "✅ Estructura del proyecto verificada" "$GREEN"
echo ""

# Verificar entorno CUDA
print_color "🔧 Verificando entorno CUDA..." "$CYAN"

# Verificar NVCC
if ! check_command "nvcc"; then
    print_color "  Instale CUDA Toolkit: https://developer.nvidia.com/cuda-downloads" "$YELLOW"
    exit 1
fi

# Verificar versión de CUDA
CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9.]+" | head -1)
print_color "  CUDA Toolkit: v$CUDA_VERSION" "$GREEN"

# Verificar GPU y determinar arquitectura
ARCH_COMPUTE=""  # Compute capability (virtual)
ARCH_CODE=""     # Real architecture code
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1)
    GPU_MEM=$(echo "$GPU_INFO" | cut -d',' -f2)
    print_color "  GPU: $GPU_NAME (${GPU_MEM} MB VRAM)" "$GREEN"

    # Ajustar arquitecturas según la GPU
    case "$GPU_NAME" in
        *"GTX 1650"*)
            ARCH_COMPUTE="compute_75"  # Virtual architecture
            ARCH_CODE="sm_75"          # Real architecture
            print_color "  ⚙️  Arquitectura detectada: Turing (compute_75, sm_75)" "$CYAN"
            ;;
        *"RTX 20"*)
            ARCH_COMPUTE="compute_75"
            ARCH_CODE="sm_75"
            print_color "  ⚙️  Arquitectura detectada: Turing (compute_75, sm_75)" "$CYAN"
            ;;
        *"RTX 30"*)
            ARCH_COMPUTE="compute_86"
            ARCH_CODE="sm_86"
            print_color "  ⚙️  Arquitectura detectada: Ampere (compute_86, sm_86)" "$CYAN"
            ;;
        *"RTX 40"*)
            ARCH_COMPUTE="compute_89"
            ARCH_CODE="sm_89"
            print_color "  ⚙️  Arquitectura detectada: Ada Lovelace (compute_89, sm_89)" "$CYAN"
            ;;
        *)
            ARCH_COMPUTE="compute_61"  # Predeterminado
            ARCH_CODE="sm_61"
            print_color "  ⚙️  Arquitectura: Predeterminada (compute_61, sm_61)" "$YELLOW"
            ;;
    esac
else
    print_color "  ⚠️  No se pudo detectar GPU NVIDIA" "$YELLOW"
    ARCH_COMPUTE="compute_61"
    ARCH_CODE="sm_61"
fi

# Verificar compilador host
GCC_VERSIONS=("gcc-14" "gcc-13" "gcc-12" "gcc-11" "gcc-10" "gcc-9" "gcc")
GCC_CMD=""
for gcc in "${GCC_VERSIONS[@]}"; do
    if command -v "$gcc" &> /dev/null; then
        GCC_CMD="$gcc"
        GCC_VERSION=$("$gcc" --version | head -n1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
        print_color "  Compilador host: $GCC_CMD v$GCC_VERSION" "$GREEN"
        break
    fi
done

if [ -z "$GCC_CMD" ]; then
    print_color "✗ Error: No se encontró ningún compilador GCC" "$RED"
    exit 1
fi

# Crear directorios necesarios
print_color "📁 Creando estructura de directorios..." "$CYAN"
mkdir -p bin
mkdir -p results
mkdir -p logs/compilation
mkdir -p logs/performance

# Generar nombre de archivo de log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/compilation/compile_${TIMESTAMP}.log"
BENCHMARK_LOG="logs/performance/benchmark_${TIMESTAMP}.log"

# Archivo de salida
OUTPUT_BINARY="bin/full_pi_search_enhanced"

print_color "🎯 Configuración de compilación:" "$MAGENTA"
echo "  • Proyecto:     $(pwd)"
echo "  • Archivo CUDA: src/cuda/full_pi_search_enhanced.cu"
echo "  • Salida:       $OUTPUT_BINARY"
echo "  • Arquitectura: $ARCH_COMPUTE ($ARCH_CODE)"
echo "  • Compilador:   $GCC_CMD"
echo "  • Log:          $LOG_FILE"
if [ $DEBUG_MODE -eq 1 ]; then
    echo "  • Modo:         Debug"
fi
if [ $BENCHMARK_MODE -eq 1 ]; then
    echo "  • Benchmark:    Sí"
fi
echo ""

# Parámetros de optimización avanzados
print_color "⚡ Configurando optimizaciones avanzadas..." "$CYAN"

# Opciones comunes
COMMON_FLAGS="-lcudart -lstdc++"

# Opciones de host compiler
HOST_FLAGS="-ccbin /usr/bin/$GCC_CMD"

# Optimizaciones NVCC
NVCC_OPTIMIZATIONS="-O3 -Xptxas -O3"

# Optimizaciones específicas de arquitectura
ARCH_OPTIMIZATIONS="-arch=$ARCH_COMPUTE -code=$ARCH_CODE"

# Generación de información de depuración
if [ $DEBUG_MODE -eq 1 ]; then
    DEBUG_FLAGS="-G -lineinfo"
    print_color "  🔍 Compilando con símbolos de depuración" "$YELLOW"
else
    DEBUG_FLAGS="--generate-line-info"
fi

# Opciones de rendimiento
PERFORMANCE_FLAGS="--ftz=true --prec-div=false --prec-sqrt=false --fmad=true"

# Opciones de código
CODE_FLAGS="--std=c++17 --extended-lambda --expt-relaxed-constexpr"

# Opciones del compilador host
XCOMPILER_FLAGS="-fopenmp -march=native -mtune=native -pipe -fno-strict-aliasing"

# Compilar
print_color "🔨 Compilando con optimizaciones avanzadas..." "$CYAN"
echo ""

START_TIME=$(date +%s.%N)

# Construir comando de compilación
NVCC_CMD="nvcc"
NVCC_CMD="$NVCC_CMD -o \"$OUTPUT_BINARY\""
NVCC_CMD="$NVCC_CMD \"src/cuda/full_pi_search_enhanced.cu\""
NVCC_CMD="$NVCC_CMD -lcudart -lstdc++"
NVCC_CMD="$NVCC_CMD -ccbin /usr/bin/$GCC_CMD"
NVCC_CMD="$NVCC_CMD -Xcompiler \"$XCOMPILER_FLAGS\""
NVCC_CMD="$NVCC_CMD -O3"
NVCC_CMD="$NVCC_CMD -Xptxas -O3"
NVCC_CMD="$NVCC_CMD -arch=$ARCH_COMPUTE"
NVCC_CMD="$NVCC_CMD -code=$ARCH_CODE"
NVCC_CMD="$NVCC_CMD $DEBUG_FLAGS"
NVCC_CMD="$NVCC_CMD --ftz=true"
NVCC_CMD="$NVCC_CMD --prec-div=false"
NVCC_CMD="$NVCC_CMD --prec-sqrt=false"
NVCC_CMD="$NVCC_CMD --fmad=true"
NVCC_CMD="$NVCC_CMD --std=c++17"
NVCC_CMD="$NVCC_CMD --extended-lambda"
NVCC_CMD="$NVCC_CMD --expt-relaxed-constexpr"
NVCC_CMD="$NVCC_CMD --resource-usage"
NVCC_CMD="$NVCC_CMD --ptxas-options=\"-v\""
NVCC_CMD="$NVCC_CMD --maxrregcount=64"
NVCC_CMD="$NVCC_CMD --use_fast_math"
# Añadir estas opciones adicionales para mejor rendimiento
NVCC_CMD="$NVCC_CMD --default-stream per-thread"
NVCC_CMD="$NVCC_CMD --extra-device-vectorization"

print_color "📝 Comando de compilación:" "$CYAN"
echo "nvcc -o \"$OUTPUT_BINARY\" \"src/cuda/full_pi_search_enhanced.cu\" \\"
echo "  -lcudart -lstdc++ \\"
echo "  -ccbin /usr/bin/$GCC_CMD \\"
echo "  -Xcompiler \"$XCOMPILER_FLAGS\" \\"
echo "  -O3 -Xptxas -O3 \\"
echo "  -arch=$ARCH_COMPUTE -code=$ARCH_CODE \\"
echo "  $DEBUG_FLAGS \\"
echo "  --ftz=true --prec-div=false --prec-sqrt=false --fmad=true \\"
echo "  --std=c++17 --extended-lambda --expt-relaxed-constexpr \\"
echo "  --resource-usage \\"
echo "  --ptxas-options=\"-v\" \\"
echo "  --maxrregcount=64 \\"
echo "  --use_fast_math \\"
echo "  --default-stream per-thread \\"
echo "  --extra-device-vectorization"
echo ""

# Ejecutar compilación
print_color "🚀 Iniciando compilación..." "$BLUE"
eval $NVCC_CMD 2>&1 | tee "$LOG_FILE"

COMPILE_STATUS=$?
END_TIME=$(date +%s.%N)
COMPILE_TIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
print_color "========================================" "$BLUE"

if [ $COMPILE_STATUS -eq 0 ]; then
    print_color "✅ Compilación exitosa" "$GREEN"
    print_color "⏱️  Tiempo de compilación: ${COMPILE_TIME}s" "$CYAN"

    # Verificar archivo ejecutable
    if [ -f "$OUTPUT_BINARY" ]; then
        # Obtener información del binario
        EXE_SIZE=$(du -h "$OUTPUT_BINARY" | cut -f1)
        EXE_SIZE_BYTES=$(stat -c%s "$OUTPUT_BINARY")

        print_color "📊 Información del ejecutable:" "$CYAN"
        echo "  • Tamaño: $EXE_SIZE ($EXE_SIZE_BYTES bytes)"

        if command -v file &> /dev/null; then
            FILE_INFO=$(file "$OUTPUT_BINARY" | cut -d: -f2-)
            echo "  • Tipo: $FILE_INFO"
        fi

        # Extraer información del log de compilación
        if [ -f "$LOG_FILE" ]; then
            print_color "📈 Estadísticas de compilación:" "$CYAN"

            # Memoria de registro
            REG_USAGE=$(grep "registers" "$LOG_FILE" | tail -1 | awk '{print $4}')
            if [ -n "$REG_USAGE" ]; then
                echo "  • Registros por hilo: $REG_USAGE"
            fi

            # Memoria compartida
            SHARED_MEM=$(grep "bytes smem" "$LOG_FILE" | tail -1 | awk '{print $5}')
            if [ -n "$SHARED_MEM" ]; then
                echo "  • Memoria compartida: $SHARED_MEM bytes"
            fi

            # Memoria constante
            CONST_MEM=$(grep "bytes cmem" "$LOG_FILE" | tail -1 | awk '{print $5}')
            if [ -n "$CONST_MEM" ]; then
                echo "  • Memoria constante: $CONST_MEM bytes"
            fi
        fi

        echo ""
        print_color "🎯 Ejecutable listo en: $OUTPUT_BINARY" "$GREEN"

        # Ejecutar benchmark de rendimiento si se solicita
        if [ $BENCHMARK_MODE -eq 1 ]; then
            print_color "📊 Ejecutando benchmark de rendimiento..." "$MAGENTA"
            echo ""

            # Verificar que existan archivos de datos para benchmark
            DATASET_FILES=("datasets/Pi - Dec.txt" "datasets/e - Dec.txt")
            DATASET_FOUND=0

            for dataset in "${DATASET_FILES[@]}"; do
                if [ -f "$dataset" ]; then
                    DATASET_FOUND=1
                    print_color "🔍 Probando con: $(basename "$dataset")" "$CYAN"
                    echo "=== BENCHMARK $(basename "$dataset") ===" >> "$BENCHMARK_LOG"
                    "./$OUTPUT_BINARY" "$dataset" --benchmark 2>&1 | tee -a "$BENCHMARK_LOG"
                    echo "" >> "$BENCHMARK_LOG"
                    echo ""
                fi
            done

            if [ $DATASET_FOUND -eq 0 ]; then
                print_color "⚠️  No se encontraron archivos de datos para benchmark" "$YELLOW"
                print_color "   Creando archivo de prueba pequeño..." "$YELLOW"

                # Crear archivo de prueba pequeño
                TEST_FILE="test_digits.txt"
                echo "Creando archivo de prueba de 10MB..."
                head -c 10485760 /dev/urandom | tr -dc '0-9' > "$TEST_FILE"
                print_color "🔍 Probando con archivo generado: $TEST_FILE" "$CYAN"
                "./$OUTPUT_BINARY" "$TEST_FILE" --benchmark 2>&1 | tee -a "$BENCHMARK_LOG"
                rm "$TEST_FILE"
            fi

            if [ -f "$BENCHMARK_LOG" ]; then
                print_color "📄 Log de benchmark guardado en: $BENCHMARK_LOG" "$GREEN"
            fi
        fi

        # Preguntar si desea ejecutar
        echo ""
        read -p "¿Ejecutar el programa ahora? (s/n): " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Ss]$ ]]; then
            print_color "========================================" "$BLUE"
            print_color "🚀 Ejecutando Constant Hunter..." "$MAGENTA"
            echo ""

            # Buscar archivo de datos predeterminado
            DEFAULT_FILE=""
            for file in "datasets/Pi - Dec.txt" "datasets/e - Dec.txt"; do
                if [ -f "$file" ]; then
                    DEFAULT_FILE="$file"
                    break
                fi
            done

            if [ -n "$DEFAULT_FILE" ]; then
                print_color "📁 Usando archivo: $DEFAULT_FILE" "$CYAN"
                "./$OUTPUT_BINARY" "$DEFAULT_FILE"
            else
                print_color "⚠️  No se encontraron archivos de datos en datasets/" "$YELLOW"
                print_color "   Ejecutando sin parámetros..." "$YELLOW"
                "./$OUTPUT_BINARY"
            fi
        fi

    else
        print_color "✗ Error: El ejecutable no se generó correctamente" "$RED"
        print_color "  Verificando si hubo errores en el log..." "$YELLOW"
        if [ -f "$LOG_FILE" ]; then
            tail -20 "$LOG_FILE"
        fi
        exit 1
    fi

else
    print_color "❌ Error en la compilación" "$RED"
    print_color "⏱️  Tiempo de compilación: ${COMPILE_TIME}s" "$CYAN"

    # Mostrar errores comunes
    if [ -f "$LOG_FILE" ]; then
        print_color "📝 Últimos errores del log:" "$YELLOW"
        tail -30 "$LOG_FILE"

        print_color "📄 Revisa el archivo de log completo: $LOG_FILE" "$YELLOW"
    fi

    # Sugerencias comunes
    echo ""
    print_color "💡 Posibles soluciones:" "$CYAN"
    echo "  1. Verifica que CUDA Toolkit esté correctamente instalado"
    echo "  2. Asegúrate de tener permisos de escritura en el directorio"
    echo "  3. Revisa errores de sintaxis en el archivo .cu"

    exit 1
fi

echo ""
print_color "=== Optimización completada ===" "$MAGENTA"
print_color "🎯 Ejecutable: $OUTPUT_BINARY" "$GREEN"
print_color "📊 Logs: logs/compilation/" "$CYAN"
print_color "🚀 Para ejecutar: ./$OUTPUT_BINARY [archivo_datos]" "$BLUE"

# Limpieza automática
cleanup
