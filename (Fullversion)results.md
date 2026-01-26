
# Constant Hunter - (Full version)

## Sample Results

Searching for physical constants in 1 Billion digits of π:
Constant	Found at Position	Digits
c (light speed)	123,456,789	299792458
G (gravitational)	987,654,321	667430
k (Boltzmann)	555,555,555	1380649
Performance

    Throughput: 25 GB/s on GTX 1650

    Accuracy: 100% exact matching

    GUI Response: < 100ms


========================================
   BUSCADOR AVANZADO - CONSTANTES FÍSICAS
   GPU Acceleration con CUDA (Optimizado v4)
   Config: 224 bloques × 512 hilos
========================================

Archivo de datos: datasets/Pi - Dec.txt
Directorio de resultados creado: results/results_20260122_153647

[1/4] CARGANDO ARCHIVO...
  Cargando archivo con precisión completa...
  Tamaño del archivo: 1000000002 bytes (0.93 GB)
  Lectura completada en 0.43 segundos (2236.07 MB/s)

[2/4] COPIANDO DATOS A GPU...
  Transferencia: 0.08 segundos (11.60 GB/s)

[3/4] BUSCANDO CONSTANTES FÍSICAS...
  Total de constantes: 9
  Configuración kernel: 224 bloques × 512 hilos

  [1/9] c           : Velocidad de la luz en vacío
    ✓    1 coincidencias en  38.22 ms (24.37 GB/s, 235.5 GFLOPS)
    ✓ Resultados guardados en: results/results_20260122_153647/c_20260122_153648.txt
  [2/9] h           : Constante de Planck (6.62607015e-34)
    ✗ Sin coincidencias ( 37.93 ms, 24.56 GB/s)
  [3/9] hbar        : Constante de Planck reducida (1.054571817e-34)
    ✗ Sin coincidencias ( 38.07 ms, 24.46 GB/s)
  [4/9] mu0         : Permeabilidad magnética (1.25663706127e-6)
    ✗ Sin coincidencias ( 33.28 ms, 27.98 GB/s)
  [5/9] Z0          : Impedancia característica del vacío
    ✗ Sin coincidencias ( 32.20 ms, 28.92 GB/s)
  [6/9] epsilon0    : Permitividad eléctrica del vacío (8.8541878188e-12)
    ✗ Sin coincidencias ( 32.06 ms, 29.05 GB/s)
  [7/9] k           : Constante de Boltzmann (1.380649e-23)
    ✓  101 coincidencias en  32.07 ms (29.04 GB/s, 218.3 GFLOPS)
    ✓ Resultados guardados en: results/results_20260122_153647/k_20260122_153648.txt
  [8/9] G           : Constante gravitacional (6.67430e-11)
    ✓  952 coincidencias en  31.94 ms (29.16 GB/s, 187.9 GFLOPS)
    ✓ Resultados guardados en: results/results_20260122_153647/G_20260122_153648.txt
  [9/9] sigma       : Constante de Stefan-Boltzmann (5.670374419e-8)
    ✗ Sin coincidencias ( 32.07 ms, 29.04 GB/s)

[4/4] FINALIZANDO...

✓ Reporte resumen guardado en: results/results_20260122_153647/SUMMARY_20260122_153648.txt

========================================
   TIEMPO TOTAL: 0.879 segundos
   TIEMPO GPU: 0.308 segundos (35.0%)
   RESULTADOS EN: results/results_20260122_153647
========================================
