cmake_minimum_required(VERSION 3.18)
project(llama2_mistral LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)

# Shared, hand-written CUDA kernel library (one copy for all models in this repo)
set(KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cuda)

add_library(kernels STATIC
    ${KERNELS_DIR}/matmul.cu
    ${KERNELS_DIR}/embedding.cu
    ${KERNELS_DIR}/rmsnorm.cu
    ${KERNELS_DIR}/rope.cu
    ${KERNELS_DIR}/flash_attention.cu
    ${KERNELS_DIR}/attention.cu
    ${KERNELS_DIR}/swiglu.cu
    ${KERNELS_DIR}/softmax.cu
    ${KERNELS_DIR}/crossentropy.cu
    ${KERNELS_DIR}/residual.cu
    ${KERNELS_DIR}/adamw.cu
)

target_include_directories(kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${KERNELS_DIR}
)

target_compile_options(kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

target_link_libraries(kernels PUBLIC CUDA::cublas CUDA::cudart)

# Flash attention is a work in progress (kernels in ../cuda/flash_attention.cu are
# still being implemented). Default OFF so the standard-attention path builds and
# trains out of the box; flip ON to experiment: cmake .. -DUSE_FLASH_ATTN=ON
option(USE_FLASH_ATTN "Use fused flash-attention kernels (WIP)" OFF)

if(USE_FLASH_ATTN)
    target_compile_definitions(kernels PUBLIC USE_FLASH_ATTN)
endif()

add_executable(train
    src/main.cpp
    src/model.cpp
    src/dataloader.cpp
)

target_include_directories(train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${KERNELS_DIR}
)

target_link_libraries(train PRIVATE kernels)

target_compile_options(train PRIVATE -O3 -Wall -Wextra)
