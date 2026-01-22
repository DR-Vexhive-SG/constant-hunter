# Constant Hunter - Documentación Técnica

## Binarios Compilados

### 1. full_pi_search_enhanced
- **Propósito**: Búsqueda principal optimizada
- **Optimizaciones**:
  - Memoria compartida para patrones
  - Desenrollado manual (unrolling)
  - Restrict pointers para alias
  - Comparación en bloques de 4
  - Configuración óptima GTX 1650

### 2. search_kernel_test
- **Propósito**: Pruebas y desarrollo
- **Características**: Kernel básico para comparaciones

## Uso

### Búsqueda completa:
```bash
./full_pi_search_enhanced "../datasets/Pi - Dec.txt"

Benchmark:
./run_benchmark.sh

Prueba rápida:
./quick_test.sh

Parámetros de Compilación

    -arch=sm_75: Para Turing architecture (GTX 1650)

    -O3: Máxima optimización

    --use_fast_math: Matemática rápida (precisión suficiente)

    --fmad=true: Fused multiply-add habilitado

    -lineinfo: Información para profiling

Rendimiento Esperado

    Throughput objetivo: 30-32 GB/s

    Precisión: Exacta (string matching)

    Memoria: Pinned memory para transferencias rápidas

Troubleshooting

    Error de memoria: Reducir tamaño de archivo o usar chunking

    Error CUDA: Verificar versión de drivers (>= 525.60)

    Rendimiento bajo: Asegurar cooling adecuado de GPU
    EOF

echo ""
echo "==========================================="
echo " COMPILACIÓN COMPLETADA"
echo "==========================================="
echo ""
echo "Binarios disponibles en: $BIN_DIR"
echo ""
echo "Para probar el sistema:"
echo " cd $BIN_DIR"
echo " ./quick_test.sh"
echo ""
echo "Para ejecutar benchmark:"
echo " cd $BIN_DIR"
echo " ./run_benchmark.sh"
echo ""
echo "Para búsqueda completa:"
echo " cd $BIN_DIR"
echo " ./full_pi_search_enhanced "../datasets/Pi - Dec.txt""
echo ""
echo "Documentación técnica: $BIN_DIR/README_TECNICO.md"


