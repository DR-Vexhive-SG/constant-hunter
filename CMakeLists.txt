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
