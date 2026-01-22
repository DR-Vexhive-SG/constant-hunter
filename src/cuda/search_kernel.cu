#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Kernel simple para pruebas y benchmark
__global__ void search_kernel_basic(const char* text, int text_len,
                                    const char* pattern, int pattern_len,
                                    int* results, int* results_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx <= text_len - pattern_len) {
        int match = 1;

        // Comparación simple (sin optimizaciones)
        for (int i = 0; i < pattern_len; i++) {
            if (text[idx + i] != pattern[i]) {
                match = 0;
                break;
            }
        }

        if (match) {
            int pos = atomicAdd(results_count, 1);
            results[pos] = idx;
        }
    }
                                    }

                                    // Función de benchmark comparativo
                                    void run_comparative_benchmark() {
                                        printf("=== BENCHMARK COMPARATIVO KERNELS ===\n\n");

                                        // Configuración de prueba
                                        const size_t test_size = 100 * 1024 * 1024; // 100MB
                                        const char* test_pattern = "123456789";
                                        int pattern_len = strlen(test_pattern);

                                        printf("Tamaño de prueba: %zu MB\n", test_size/(1024*1024));
                                        printf("Patrón: '%s' (%d dígitos)\n\n", test_pattern, pattern_len);

                                        // TODO: Implementar benchmark comparativo entre kernels
                                        printf("(Benchmark comparativo pendiente de implementación)\n");
                                        printf("Use el kernel optimizado para producción.\n\n");
                                    }

                                    int main() {
                                        printf("=== Kernel de Búsqueda Básico (Para Pruebas) ===\n\n");

                                        run_comparative_benchmark();

                                        printf("Este kernel es para:\n");
                                        printf("1. Pruebas de funcionalidad básica\n");
                                        printf("2. Comparaciones de rendimiento\n");
                                        printf("3. Desarrollo y debug\n\n");
                                        printf("Para producción, use 'full_pi_search_enhanced.cu'\n");

                                        return 0;
                                    }
