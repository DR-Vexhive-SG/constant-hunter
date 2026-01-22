#!/bin/bash
# Prueba rápida del sistema

echo "=== PRUEBA RÁPIDA CONSTANT HUNTER ==="
echo ""

# Verificar GPU
echo "1. Verificando GPU CUDA..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "2. Verificando binarios CUDA..."
if [ -f "./full_pi_search_enhanced" ]; then
    echo "✓ full_pi_search_enhanced encontrado"
    # Mostrar información del binario
    echo "   Información:"
    strings ./full_pi_search_enhanced | grep -E "(sm_|cuda|compute)" | head -5
else
    echo "✗ full_pi_search_enhanced NO encontrado"
fi

echo ""
echo "3. Verificando datasets..."
if [ -d "../datasets" ]; then
    file_count=$(find ../datasets -name "*.txt" -type f | wc -l)
    echo "✓ Directorio datasets encontrado ($file_count archivos)"

    # Listar archivos con tamaño
    echo "   Archivos disponibles:"
    find ../datasets -name "*.txt" -type f -exec sh -c '
        file="$1"
        size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file")
        size_mb=$(echo "scale=1; $size / (1024*1024)" | bc)
        echo "     - $(basename "$file") (${size_mb} MB)"
    ' _ {} \;
else
    echo "✗ Directorio datasets NO encontrado"
fi

echo ""
echo "4. Para ejecutar búsqueda completa:"
echo "   ./full_pi_search_enhanced \"../datasets/Pi - Dec.txt\""
echo ""
echo "5. Para ejecutar benchmark:"
echo "   ./run_benchmark.sh"
