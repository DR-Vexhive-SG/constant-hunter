// full_pi_search_enhanced.cu - VERSIÓN CORREGIDA CON TRANSFERENCIAS ÓPTIMAS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

// =====================
// CONFIGURACIÓN OPTIMIZADA
// =====================
#define MAX_RESULTS_PER_CONSTANT 1000000
#define TIMESTAMP_BUFFER_SIZE 64
#define MAX_CONSTANT_NAME_LEN 32
#define MAX_DESCRIPTION_LEN 128
#define MAX_FILENAME_LEN 256
#define RESULTS_BASE_DIR "results"

// Configuración óptima según benchmark
#define OPTIMAL_THREADS_PER_BLOCK 512
#define OPTIMAL_BLOCKS_PER_GRID 224

// Configuración de memoria compartida optimizada
#define SHARED_PATTERN_SIZE 64
#define PREFETCH_DISTANCE 128

// =====================
// KERNEL DE BÚSQUEDA SUPER OPTIMIZADO V4
// =====================
__global__ void search_kernel_optimized_v4(const char* __restrict__ text,
                                           long long text_len,
                                           const char* __restrict__ pattern,
                                           int pattern_len,
                                           int* __restrict__ results,
                                           int* __restrict__ results_count) {
    // Índice global optimizado
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    // Memoria compartida para patrón
    __shared__ char shared_pattern[SHARED_PATTERN_SIZE];

    // Carga coalescente del patrón
    int load_size = (pattern_len + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < load_size; i++) {
        int pos = threadIdx.x + i * blockDim.x;
        if (pos < pattern_len && pos < SHARED_PATTERN_SIZE) {
            shared_pattern[pos] = pattern[pos];
        }
    }
    __syncthreads();

    // Búsqueda principal optimizada
    for (long long i = idx; i <= text_len - pattern_len; i += stride) {
        bool match = true;

        // Optimización basada en tamaño del patrón
        if (pattern_len <= 16) {
            // Desenrollado completo para patrones pequeños
            #pragma unroll
            for (int j = 0; j < pattern_len; j++) {
                if (text[i + j] != shared_pattern[j]) {
                    match = false;
                    break;
                }
            }
        } else {
            // Búsqueda por bloques con mejor coalescencia
            int j = 0;
            while (j < pattern_len && match) {
                if (j + 7 < pattern_len) {
                    // Comparar 8 bytes a la vez (óptimo para coalescencia)
                    if (text[i + j] != shared_pattern[j] ||
                        text[i + j + 1] != shared_pattern[j + 1] ||
                        text[i + j + 2] != shared_pattern[j + 2] ||
                        text[i + j + 3] != shared_pattern[j + 3] ||
                        text[i + j + 4] != shared_pattern[j + 4] ||
                        text[i + j + 5] != shared_pattern[j + 5] ||
                        text[i + j + 6] != shared_pattern[j + 6] ||
                        text[i + j + 7] != shared_pattern[j + 7]) {
                        match = false;
                    break;
                        }
                        j += 8;
                } else {
                    if (text[i + j] != shared_pattern[j]) {
                        match = false;
                        break;
                    }
                    j++;
                }
            }
        }

        if (match) {
            int pos = atomicAdd(results_count, 1);
            if (pos < MAX_RESULTS_PER_CONSTANT) {
                results[pos] = i;
            }
        }
    }
                                           }

                                           // =====================
                                           // CONSTANTES FÍSICAS
                                           // =====================
                                           struct PhysicalConstant {
                                               const char* name;
                                               const char* digits;
                                               const char* description;
                                           };

                                           PhysicalConstant constants[] = {
                                               {"c",        "299792458",     "Velocidad de la luz en vacío"},
                                               {"h",        "662607015",     "Constante de Planck (6.62607015e-34)"},
                                               {"hbar",     "1054571817",    "Constante de Planck reducida (1.054571817e-34)"},
                                               {"mu0",      "125663706127",  "Permeabilidad magnética (1.25663706127e-6)"},
                                               {"Z0",       "376730313412",  "Impedancia característica del vacío"},
                                               {"epsilon0", "88541878188",   "Permitividad eléctrica del vacío (8.8541878188e-12)"},
                                               {"k",        "1380649",       "Constante de Boltzmann (1.380649e-23)"},
                                               {"G",        "667430",        "Constante gravitacional (6.67430e-11)"},
                                               {"sigma",    "5670374419",    "Constante de Stefan-Boltzmann (5.670374419e-8)"}
                                           };

                                           #define NUM_CONSTANTS (sizeof(constants)/sizeof(constants[0]))

                                           // =====================
                                           // ESTRUCTURA PARA RESULTADOS
                                           // =====================
                                           struct SearchResult {
                                               char name[MAX_CONSTANT_NAME_LEN];
                                               char description[MAX_DESCRIPTION_LEN];
                                               const char* digits;
                                               int pattern_len;
                                               int match_count;
                                               float search_time_ms;
                                               float throughput_gbs;
                                               float gflops;
                                           };

                                           // =====================
                                           // UTILIDADES
                                           // =====================
                                           #define CHECK_CUDA_ERROR(call) {                                         \
                                           cudaError_t err = call;                                              \
                                           if (err != cudaSuccess) {                                            \
                                               fprintf(stderr, "CUDA Error en %s:%d: %s\n",                     \
                                               __FILE__, __LINE__, cudaGetErrorString(err));            \
                                               exit(EXIT_FAILURE);                                              \
                                           }                                                                    \
                                           }

                                           void extract_base_filename(const char* fullpath, char* basename, size_t size) {
                                               const char* start = strrchr(fullpath, '/');
                                               if (!start) start = strrchr(fullpath, '\\');
                                               if (!start) start = fullpath;
                                               else start++;

                                               const char* end = strrchr(start, '.');
                                               size_t len = end ? (size_t)(end - start) : strlen(start);

                                               if (len > size - 1) len = size - 1;
                                               strncpy(basename, start, len);
                                               basename[len] = '\0';
                                           }

                                           char* create_results_directory() {
                                               struct stat st = {0};
                                               if (stat(RESULTS_BASE_DIR, &st) == -1) {
                                                   mkdir(RESULTS_BASE_DIR, 0755);
                                               }

                                               char timestamp[TIMESTAMP_BUFFER_SIZE];
                                               time_t rawtime = time(NULL);
                                               strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&rawtime));

                                               static char dir_path[256];
                                               snprintf(dir_path, sizeof(dir_path), "%s/results_%s", RESULTS_BASE_DIR, timestamp);

                                               mkdir(dir_path, 0755);
                                               printf("Directorio de resultados creado: %s\n", dir_path);

                                               return dir_path;
                                           }

                                           // =====================
                                           // CARGA DE ARCHIVOS OPTIMIZADA
                                           // =====================
                                           char* load_file_optimized(const char* filename, long long* file_size, double* load_time) {
                                               printf("  Cargando archivo con precisión completa...\n");

                                               clock_t start = clock();

                                               FILE* file = fopen(filename, "rb");
                                               if (!file) {
                                                   fprintf(stderr, "Error abriendo archivo: %s\n", filename);
                                                   return NULL;
                                               }

                                               fseek(file, 0, SEEK_END);
                                               *file_size = ftell(file);
                                               fseek(file, 0, SEEK_SET);

                                               printf("  Tamaño del archivo: %lld bytes (%.2f GB)\n",
                                                      *file_size, (double)*file_size/(1024*1024*1024));

                                               // Reservar memoria pinned para transferencia rápida a GPU
                                               char* buffer = NULL;
                                               CHECK_CUDA_ERROR(cudaMallocHost((void**)&buffer, *file_size));

                                               // Leer archivo en bloques grandes para optimizar I/O
                                               const size_t BLOCK_SIZE = 64 * 1024 * 1024; // 64MB
                                               size_t remaining = *file_size;
                                               char* ptr = buffer;

                                               while (remaining > 0) {
                                                   size_t to_read = (remaining > BLOCK_SIZE) ? BLOCK_SIZE : remaining;
                                                   size_t read = fread(ptr, 1, to_read, file);
                                                   if (read != to_read) {
                                                       fprintf(stderr, "Error de lectura\n");
                                                       cudaFreeHost(buffer);
                                                       fclose(file);
                                                       return NULL;
                                                   }
                                                   ptr += read;
                                                   remaining -= read;
                                               }

                                               fclose(file);

                                               clock_t end = clock();
                                               *load_time = (double)(end - start) / CLOCKS_PER_SEC;

                                               printf("  Lectura completada en %.2f segundos (%.2f MB/s)\n",
                                                      *load_time, (*file_size/(1024.0*1024.0)) / *load_time);

                                               return buffer;
                                           }

                                           // =====================
                                           // FUNCIONES DE RESULTADOS
                                           // =====================
                                           void save_results_with_timestamp(const char* results_dir,
                                                                            const struct SearchResult* result,
                                                                            const struct PhysicalConstant* constant,
                                                                            int* positions, long long file_size,
                                                                            const char* data_filename) {
                                               char filename[MAX_FILENAME_LEN];
                                               char timestamp[TIMESTAMP_BUFFER_SIZE];
                                               char data_basename[MAX_FILENAME_LEN];

                                               time_t rawtime = time(NULL);
                                               strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&rawtime));

                                               extract_base_filename(data_filename, data_basename, sizeof(data_basename));

                                               snprintf(filename, sizeof(filename), "%s/%s_%s.txt",
                                                        results_dir, result->name, timestamp);

                                               FILE* file = fopen(filename, "w");
                                               if (!file) {
                                                   perror("Error creando archivo de resultados");
                                                   return;
                                               }

                                               fprintf(file, "# ========================================\n");
                                               fprintf(file, "# RESULTADOS DE BÚSQUEDA EN %s\n", data_basename);
                                               fprintf(file, "# ========================================\n");
                                               fprintf(file, "# Constante: %s\n", result->name);
                                               fprintf(file, "# Descripción: %s\n", result->description);
                                               fprintf(file, "# Dígitos buscados: ");

                                               for (int i = 0; i < result->pattern_len; i++) {
                                                   fprintf(file, "%c", constant->digits[i]);
                                                   if ((i+1) % 10 == 0 && (i+1) < result->pattern_len) {
                                                       fprintf(file, " ");
                                                   }
                                               }
                                               fprintf(file, "\n");

                                               fprintf(file, "# Longitud del patrón: %d dígitos\n", result->pattern_len);
                                               fprintf(file, "# Coincidencias encontradas: %d\n", result->match_count);
                                               fprintf(file, "# Tiempo de búsqueda: %.2f ms\n", result->search_time_ms);
                                               fprintf(file, "# Throughput: %.2f GB/s\n", result->throughput_gbs);
                                               if (result->gflops > 0) {
                                                   fprintf(file, "# Rendimiento: %.1f GFLOPS\n", result->gflops);
                                               }
                                               fprintf(file, "# Tamaño archivo: %lld bytes (%.2f GB)\n",
                                                       file_size, (double)file_size/(1024*1024*1024));

                                               char time_str[TIMESTAMP_BUFFER_SIZE];
                                               strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&rawtime));
                                               fprintf(file, "# Timestamp análisis: %s\n", time_str);
                                               fprintf(file, "# ========================================\n\n");

                                               fprintf(file, "POSICIONES ENCONTRADAS (offset desde inicio):\n");
                                               fprintf(file, "----------------------------------------\n");

                                               int display_limit = (result->match_count > 1000) ? 1000 : result->match_count;
                                               for (int i = 0; i < display_limit; i++) {
                                                   fprintf(file, "%d\n", positions[i]);
                                               }

                                               if (result->match_count > 1000) {
                                                   fprintf(file, "... (y %d posiciones más)\n", result->match_count - 1000);
                                               }

                                               fclose(file);
                                               printf("    ✓ Resultados guardados en: %s\n", filename);
                                                                            }

                                                                            void generate_summary_report(const char* results_dir,
                                                                                                         const struct SearchResult* results,
                                                                                                         int num_constants,
                                                                                                         double total_time_seconds,
                                                                                                         double data_load_time,
                                                                                                         double gpu_copy_time,
                                                                                                         long long file_size,
                                                                                                         const char* data_filename) {
                                                                                char filename[MAX_FILENAME_LEN];
                                                                                char timestamp[TIMESTAMP_BUFFER_SIZE];
                                                                                char data_basename[MAX_FILENAME_LEN];

                                                                                time_t rawtime = time(NULL);
                                                                                strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&rawtime));

                                                                                extract_base_filename(data_filename, data_basename, sizeof(data_basename));

                                                                                snprintf(filename, sizeof(filename), "%s/SUMMARY_%s.txt", results_dir, timestamp);

                                                                                FILE* file = fopen(filename, "w");
                                                                                if (!file) {
                                                                                    perror("Error creando reporte resumen");
                                                                                    return;
                                                                                }

                                                                                fprintf(file, "========================================\n");
                                                                                fprintf(file, "   RESUMEN DE ANÁLISIS - CONSTANTES EN %s\n", data_basename);
                                                                                fprintf(file, "========================================\n\n");

                                                                                char time_str[TIMESTAMP_BUFFER_SIZE];
                                                                                strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&rawtime));
                                                                                fprintf(file, "FECHA Y HORA DEL ANÁLISIS: %s\n", time_str);
                                                                                fprintf(file, "ARCHIVO ANALIZADO: %s\n", data_filename);
                                                                                fprintf(file, "TAMAÑO: %lld bytes (%.2f GB)\n", file_size, (double)file_size/(1024*1024*1024));

                                                                                cudaDeviceProp prop;
                                                                                cudaGetDeviceProperties(&prop, 0);
                                                                                fprintf(file, "GPU UTILIZADA: %s (%d multiprocesadores, %.1f GB VRAM)\n\n",
                                                                                        prop.name, prop.multiProcessorCount,
                                                                                        (double)prop.totalGlobalMem/(1024*1024*1024));

                                                                                fprintf(file, "TIEMPOS DE EJECUCIÓN:\n");
                                                                                fprintf(file, "----------------------------------------\n");
                                                                                fprintf(file, "  • Carga de archivo:       %.3f segundos\n", data_load_time);
                                                                                fprintf(file, "  • Copia a GPU:            %.3f segundos\n", gpu_copy_time);
                                                                                fprintf(file, "  • Búsqueda total:         %.3f segundos\n",
                                                                                        total_time_seconds - data_load_time - gpu_copy_time);
                                                                                fprintf(file, "  • TIEMPO TOTAL:           %.3f segundos\n", total_time_seconds);

                                                                                double throughput_avg = (file_size/(1024.0*1024.0*1024.0)) / total_time_seconds;
                                                                                fprintf(file, "  • Throughput promedio:    %.2f GB/s\n", throughput_avg);

                                                                                double total_operations = (double)file_size * num_constants;
                                                                                double gflops = (total_operations / 1e9) / total_time_seconds;
                                                                                fprintf(file, "  • Rendimiento estimado:   %.1f GFLOPS\n", gflops);
                                                                                fprintf(file, "\n");

                                                                                int total_matches = 0;
                                                                                int constants_found = 0;
                                                                                float total_search_time = 0;
                                                                                float max_throughput = 0;
                                                                                float min_throughput = 1e9;

                                                                                fprintf(file, "RESULTADOS POR CONSTANTE:\n");
                                                                                fprintf(file, "----------------------------------------\n");

                                                                                for (int i = 0; i < num_constants; i++) {
                                                                                    const struct SearchResult* r = &results[i];
                                                                                    fprintf(file, "  %-12s: %4d coincidencias  (%.2f ms, %.2f GB/s)",
                                                                                            r->name, r->match_count, r->search_time_ms, r->throughput_gbs);

                                                                                    if (r->match_count > 0) {
                                                                                        fprintf(file, "  ✓");
                                                                                    }
                                                                                    fprintf(file, "\n");

                                                                                    total_matches += r->match_count;
                                                                                    total_search_time += r->search_time_ms;
                                                                                    if (r->match_count > 0) constants_found++;

                                                                                    if (r->throughput_gbs > max_throughput) max_throughput = r->throughput_gbs;
                                                                                    if (r->throughput_gbs < min_throughput) min_throughput = r->throughput_gbs;
                                                                                }

                                                                                fprintf(file, "\n");
                                                                                fprintf(file, "ESTADÍSTICAS GLOBALES:\n");
                                                                                fprintf(file, "----------------------------------------\n");
                                                                                fprintf(file, "  • Constantes buscadas:    %d\n", num_constants);
                                                                                fprintf(file, "  • Constantes encontradas: %d (%.1f%%)\n",
                                                                                        constants_found, (100.0 * constants_found) / num_constants);
                                                                                fprintf(file, "  • Total coincidencias:    %d\n", total_matches);
                                                                                fprintf(file, "  • Tiempo búsqueda avg:    %.2f ms/constante\n",
                                                                                        total_search_time / num_constants);
                                                                                fprintf(file, "  • Throughput máximo:      %.2f GB/s\n", max_throughput);
                                                                                fprintf(file, "  • Throughput mínimo:      %.2f GB/s\n", min_throughput);
                                                                                fprintf(file, "\n");

                                                                                if (total_matches > 0) {
                                                                                    fprintf(file, "DISTRIBUCIÓN ESPACIAL APROXIMADA:\n");
                                                                                    fprintf(file, "----------------------------------------\n");
                                                                                    fprintf(file, "  • Densidad: 1 coincidencia cada %.1f MB\n",
                                                                                            (double)file_size / (1024.0*1024.0) / total_matches);
                                                                                }

                                                                                fprintf(file, "\n========================================\n");
                                                                                fprintf(file, "   ANÁLISIS COMPLETADO EXITOSAMENTE\n");
                                                                                fprintf(file, "========================================\n");

                                                                                fclose(file);
                                                                                printf("\n✓ Reporte resumen guardado en: %s\n", filename);
                                                                                                         }

                                                                                                         // =====================
                                                                                                         // BENCHMARK DE RENDIMIENTO
                                                                                                         // =====================
                                                                                                         void run_benchmark(const char* data_filename) {
                                                                                                             printf("\n=== BENCHMARK DE RENDIMIENTO ===\n");

                                                                                                             long long file_size = 0;
                                                                                                             double load_time = 0;
                                                                                                             char* h_text = load_file_optimized(data_filename, &file_size, &load_time);
                                                                                                             if (!h_text) return;

                                                                                                             // Transferencia a GPU
                                                                                                             char* d_text = NULL;
                                                                                                             clock_t copy_start = clock();
                                                                                                             CHECK_CUDA_ERROR(cudaMalloc((void**)&d_text, file_size));
                                                                                                             CHECK_CUDA_ERROR(cudaMemcpy(d_text, h_text, file_size, cudaMemcpyHostToDevice));
                                                                                                             clock_t copy_end = clock();
                                                                                                             double copy_time = (double)(copy_end - copy_start) / CLOCKS_PER_SEC;

                                                                                                             printf("  Transferencia H2D: %.3f segundos (%.2f GB/s)\n",
                                                                                                                    copy_time, (file_size/(1024.0*1024.0*1024.0))/copy_time);

                                                                                                             // Patrón de prueba
                                                                                                             const char* test_pattern = "123456789";
                                                                                                             int pattern_len = strlen(test_pattern);
                                                                                                             char* d_pattern = NULL;
                                                                                                             CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern, pattern_len + 1));
                                                                                                             CHECK_CUDA_ERROR(cudaMemcpy(d_pattern, test_pattern, pattern_len + 1, cudaMemcpyHostToDevice));

                                                                                                             // Memoria para resultados
                                                                                                             int* d_results = NULL;
                                                                                                             int* d_results_count = NULL;
                                                                                                             CHECK_CUDA_ERROR(cudaMalloc((void**)&d_results, MAX_RESULTS_PER_CONSTANT * sizeof(int)));
                                                                                                             CHECK_CUDA_ERROR(cudaMalloc((void**)&d_results_count, sizeof(int)));

                                                                                                             // Ejecutar benchmark con diferentes configuraciones
                                                                                                             printf("\n  Probando configuraciones de kernel:\n");

                                                                                                             struct KernelConfig {
                                                                                                                 int threads;
                                                                                                                 int blocks;
                                                                                                             } configs[] = {
                                                                                                                 {128, 896},  // Configuración 1
                                                                                                                 {256, 448},  // Configuración 2 (anterior óptima)
                                                                                                                 {256, 896},  // Configuración 3
                                                                                                                 {512, 224},  // Configuración 4 (nueva óptima según benchmark)
                                                                                                             };

                                                                                                             for (int i = 0; i < 4; i++) {
                                                                                                                 CHECK_CUDA_ERROR(cudaMemset(d_results_count, 0, sizeof(int)));

                                                                                                                 cudaEvent_t start, stop;
                                                                                                                 CHECK_CUDA_ERROR(cudaEventCreate(&start));
                                                                                                                 CHECK_CUDA_ERROR(cudaEventCreate(&stop));

                                                                                                                 CHECK_CUDA_ERROR(cudaEventRecord(start));
                                                                                                                 search_kernel_optimized_v4<<<configs[i].blocks, configs[i].threads>>>(
                                                                                                                     d_text, file_size, d_pattern, pattern_len, d_results, d_results_count
                                                                                                                 );
                                                                                                                 CHECK_CUDA_ERROR(cudaEventRecord(stop));
                                                                                                                 CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

                                                                                                                 float ms;
                                                                                                                 CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

                                                                                                                 int matches;
                                                                                                                 CHECK_CUDA_ERROR(cudaMemcpy(&matches, d_results_count, sizeof(int), cudaMemcpyDeviceToHost));

                                                                                                                 double throughput = (file_size/(1024.0*1024.0*1024.0)) / (ms / 1000.0);
                                                                                                                 printf("    Config %d: %d blk × %d thr = %.2f ms (%.2f GB/s) [%d match]\n",
                                                                                                                        i+1, configs[i].blocks, configs[i].threads, ms, throughput, matches);

                                                                                                                 CHECK_CUDA_ERROR(cudaEventDestroy(start));
                                                                                                                 CHECK_CUDA_ERROR(cudaEventDestroy(stop));
                                                                                                             }

                                                                                                             // Limpiar
                                                                                                             CHECK_CUDA_ERROR(cudaFreeHost(h_text));
                                                                                                             CHECK_CUDA_ERROR(cudaFree(d_text));
                                                                                                             CHECK_CUDA_ERROR(cudaFree(d_pattern));
                                                                                                             CHECK_CUDA_ERROR(cudaFree(d_results));
                                                                                                             CHECK_CUDA_ERROR(cudaFree(d_results_count));

                                                                                                             printf("\n=== BENCHMARK COMPLETADO ===\n");
                                                                                                         }

                                                                                                         // =====================
                                                                                                         // PROGRAMA PRINCIPAL
                                                                                                         // =====================
                                                                                                         int main(int argc, char** argv) {
                                                                                                             printf("\n========================================\n");
                                                                                                             printf("   BUSCADOR AVANZADO - CONSTANTES FÍSICAS\n");
                                                                                                             printf("   GPU Acceleration con CUDA (Optimizado v4)\n");
                                                                                                             printf("   Config: %d bloques × %d hilos\n", OPTIMAL_BLOCKS_PER_GRID, OPTIMAL_THREADS_PER_BLOCK);
                                                                                                             printf("========================================\n\n");

                                                                                                             clock_t program_start = clock();

                                                                                                             // Determinar archivo de entrada
                                                                                                             const char* data_file = (argc > 1) ? argv[1] : "datasets/Pi - Dec.txt";

                                                                                                             // Opción de benchmark
                                                                                                             if (argc > 2 && strcmp(argv[2], "--benchmark") == 0) {
                                                                                                                 run_benchmark(data_file);
                                                                                                                 return 0;
                                                                                                             }

                                                                                                             printf("Archivo de datos: %s\n", data_file);

                                                                                                             // 1. Crear directorio de resultados
                                                                                                             char* results_dir = create_results_directory();

                                                                                                             // 2. Cargar archivo optimizado con memoria pinned
                                                                                                             printf("\n[1/4] CARGANDO ARCHIVO...\n");
                                                                                                             long long file_size = 0;
                                                                                                             double load_time = 0;
                                                                                                             char* h_text = load_file_optimized(data_file, &file_size, &load_time);
                                                                                                             if (!h_text) return 1;

                                                                                                             // 3. Copiar a GPU (h_text ya es pinned memory)
                                                                                                             printf("\n[2/4] COPIANDO DATOS A GPU...\n");
                                                                                                             char* d_text = NULL;
                                                                                                             clock_t copy_start = clock();
                                                                                                             CHECK_CUDA_ERROR(cudaMalloc((void**)&d_text, file_size));
                                                                                                             CHECK_CUDA_ERROR(cudaMemcpy(d_text, h_text, file_size, cudaMemcpyHostToDevice));
                                                                                                             clock_t copy_end = clock();
                                                                                                             double copy_time = (double)(copy_end - copy_start) / CLOCKS_PER_SEC;

                                                                                                             printf("  Transferencia: %.2f segundos (%.2f GB/s)\n",
                                                                                                                    copy_time, (file_size/(1024.0*1024.0*1024.0))/copy_time);

                                                                                                             // 4. Buscar constantes
                                                                                                             printf("\n[3/4] BUSCANDO CONSTANTES FÍSICAS...\n");
                                                                                                             printf("  Total de constantes: %zu\n", NUM_CONSTANTS);
                                                                                                             printf("  Configuración kernel: %d bloques × %d hilos\n\n",
                                                                                                                    OPTIMAL_BLOCKS_PER_GRID, OPTIMAL_THREADS_PER_BLOCK);

                                                                                                             struct SearchResult global_results[NUM_CONSTANTS];
                                                                                                             float total_gpu_time = 0;

                                                                                                             for (int c = 0; c < NUM_CONSTANTS; c++) {
                                                                                                                 printf("  [%d/%zu] %-12s: %s\n",
                                                                                                                        c+1, NUM_CONSTANTS, constants[c].name, constants[c].description);

                                                                                                                 // Preparar patrón
                                                                                                                 int pattern_len = strlen(constants[c].digits);
                                                                                                                 char* d_pattern = NULL;
                                                                                                                 CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern, pattern_len + 1));
                                                                                                                 CHECK_CUDA_ERROR(cudaMemcpy(d_pattern, constants[c].digits,
                                                                                                                                             pattern_len + 1, cudaMemcpyHostToDevice));

                                                                                                                 // Memoria para resultados
                                                                                                                 int* d_results = NULL;
                                                                                                                 int* d_results_count = NULL;
                                                                                                                 int* h_results = NULL;

                                                                                                                 CHECK_CUDA_ERROR(cudaMalloc((void**)&d_results,
                                                                                                                                             MAX_RESULTS_PER_CONSTANT * sizeof(int)));
                                                                                                                 CHECK_CUDA_ERROR(cudaMalloc((void**)&d_results_count, sizeof(int)));
                                                                                                                 CHECK_CUDA_ERROR(cudaMemset(d_results_count, 0, sizeof(int)));

                                                                                                                 h_results = (int*)malloc(MAX_RESULTS_PER_CONSTANT * sizeof(int));

                                                                                                                 // Ejecutar kernel optimizado v4
                                                                                                                 cudaEvent_t start, stop;
                                                                                                                 CHECK_CUDA_ERROR(cudaEventCreate(&start));
                                                                                                                 CHECK_CUDA_ERROR(cudaEventCreate(&stop));

                                                                                                                 CHECK_CUDA_ERROR(cudaEventRecord(start));
                                                                                                                 search_kernel_optimized_v4<<<OPTIMAL_BLOCKS_PER_GRID, OPTIMAL_THREADS_PER_BLOCK>>>(
                                                                                                                     d_text, file_size, d_pattern, pattern_len, d_results, d_results_count
                                                                                                                 );
                                                                                                                 CHECK_CUDA_ERROR(cudaEventRecord(stop));
                                                                                                                 CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

                                                                                                                 float search_time_ms = 0;
                                                                                                                 CHECK_CUDA_ERROR(cudaEventElapsedTime(&search_time_ms, start, stop));
                                                                                                                 total_gpu_time += search_time_ms;

                                                                                                                 // Recuperar resultados
                                                                                                                 int match_count = 0;
                                                                                                                 CHECK_CUDA_ERROR(cudaMemcpy(&match_count, d_results_count,
                                                                                                                                             sizeof(int), cudaMemcpyDeviceToHost));

                                                                                                                 // Almacenar resultados
                                                                                                                 strncpy(global_results[c].name, constants[c].name, MAX_CONSTANT_NAME_LEN-1);
                                                                                                                 strncpy(global_results[c].description, constants[c].description,
                                                                                                                         MAX_DESCRIPTION_LEN-1);
                                                                                                                 global_results[c].pattern_len = pattern_len;
                                                                                                                 global_results[c].match_count = match_count;
                                                                                                                 global_results[c].search_time_ms = search_time_ms;

                                                                                                                 double throughput = (file_size/(1024.0*1024.0*1024.0)) / (search_time_ms / 1000.0);
                                                                                                                 global_results[c].throughput_gbs = throughput;

                                                                                                                 double operations = (double)file_size * pattern_len;
                                                                                                                 global_results[c].gflops = (operations / 1e9) / (search_time_ms / 1000.0);

                                                                                                                 if (match_count > 0) {
                                                                                                                     CHECK_CUDA_ERROR(cudaMemcpy(h_results, d_results,
                                                                                                                                                 match_count * sizeof(int),
                                                                                                                                                 cudaMemcpyDeviceToHost));

                                                                                                                     printf("    ✓ %4d coincidencias en %6.2f ms (%.2f GB/s, %.1f GFLOPS)\n",
                                                                                                                            match_count, search_time_ms, throughput, global_results[c].gflops);

                                                                                                                     // Guardar resultados
                                                                                                                     save_results_with_timestamp(results_dir, &global_results[c],
                                                                                                                                                 &constants[c], h_results, file_size, data_file);
                                                                                                                 } else {
                                                                                                                     printf("    ✗ Sin coincidencias (%6.2f ms, %.2f GB/s)\n",
                                                                                                                            search_time_ms, throughput);
                                                                                                                 }

                                                                                                                 // Limpiar
                                                                                                                 free(h_results);
                                                                                                                 CHECK_CUDA_ERROR(cudaFree(d_pattern));
                                                                                                                 CHECK_CUDA_ERROR(cudaFree(d_results));
                                                                                                                 CHECK_CUDA_ERROR(cudaFree(d_results_count));
                                                                                                                 CHECK_CUDA_ERROR(cudaEventDestroy(start));
                                                                                                                 CHECK_CUDA_ERROR(cudaEventDestroy(stop));
                                                                                                             }

                                                                                                             // 5. Finalizar
                                                                                                             printf("\n[4/4] FINALIZANDO...\n");
                                                                                                             CHECK_CUDA_ERROR(cudaFreeHost(h_text));
                                                                                                             CHECK_CUDA_ERROR(cudaFree(d_text));

                                                                                                             clock_t program_end = clock();
                                                                                                             double total_time = (double)(program_end - program_start) / CLOCKS_PER_SEC;

                                                                                                             // Generar reporte resumen
                                                                                                             generate_summary_report(results_dir, global_results, NUM_CONSTANTS,
                                                                                                                                     total_time, load_time, copy_time, file_size, data_file);

                                                                                                             printf("\n========================================\n");
                                                                                                             printf("   TIEMPO TOTAL: %.3f segundos\n", total_time);
                                                                                                             printf("   TIEMPO GPU: %.3f segundos (%.1f%%)\n",
                                                                                                                    total_gpu_time/1000.0, (total_gpu_time/1000.0)/total_time*100);
                                                                                                             printf("   RESULTADOS EN: %s\n", results_dir);
                                                                                                             printf("========================================\n\n");

                                                                                                             return 0;
                                                                                                         }
