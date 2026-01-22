#!/bin/bash
# Script de benchmark para Constant Hunter

echo "=== BENCHMARK CONSTANT HUNTER ==="
echo ""

# Verificar binarios
if [ ! -f "./full_pi_search_enhanced" ]; then
    echo "Error: full_pi_search_enhanced no encontrado"
    exit 1
fi

# Archivos de prueba disponibles
echo "Archivos disponibles para benchmark:"
find ../datasets -name "*.txt" -type f | while read file; do
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file")
    size_mb=$(echo "scale=2; $size / (1024*1024)" | bc)
    echo "  - $(basename "$file") (${size_mb} MB)"
done

echo ""
echo "Ejecutando benchmark en archivo Pi - Dec.txt..."
echo ""

# Ejecutar con opción de benchmark
if [ -f "../datasets/Pi - Dec.txt" ]; then
    ./full_pi_search_enhanced "../datasets/Pi - Dec.txt" --benchmark
else
    echo "Archivo Pi - Dec.txt no encontrado en datasets/"
    echo "Por favor coloque archivos en ../datasets/"
fi
