#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

// Wraps every CUDA call. On failure, prints file, line, and error message, then exits.
#define CUDA_CHECK(err)                                                     \
    do {                                                                    \
        cudaError_t e = (err);                                              \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(e));                                  \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Same as CUDA_CHECK but for cuBLAS calls, which return cublasStatus_t instead of cudaError_t.
#define CUBLAS_CHECK(err)                                                   \
    do {                                                                    \
        cublasStatus_t s = (err);                                           \
        if (s != CUBLAS_STATUS_SUCCESS){                                    \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__,    \
                    __LINE__, (int)s);                                       \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// GPU tensor wrapping a device pointer and its element count.
// Does NOT do reference counting. Caller is responsible for calling free().
struct Tensor {
    float* data = nullptr;  // device (GPU) pointer
    size_t numel = 0;       // number of float elements (NOT bytes)

    // Allocate n floats on the GPU. Frees any existing allocation first.
    void allocate(size_t n) {
        free();
        numel = n;
        CUDA_CHECK(cudaMalloc(&data, n * sizeof(float)));
    }

    // Release GPU memory. Sets pointer to null so we don't double-free.
    void free() {
        if (data) {
            CUDA_CHECK(cudaFree(data));
            data = nullptr;
            numel = 0;
        }
    }

    // Fill entire buffer with zeros. Used to clear gradient buffers between steps.
    void zero() {
        if (data) {
            CUDA_CHECK(cudaMemset(data, 0, numel * sizeof(float)));
        }
    }

    // Copy n floats from CPU array into this GPU tensor. Auto-allocates if too small.
    void from_host(const float* src, size_t n) {
        if (numel < n) allocate(n);
        CUDA_CHECK(cudaMemcpy(data, src, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Copy n floats from this GPU tensor into a CPU array.
    void to_host(float* dst, size_t n) const {
        CUDA_CHECK(cudaMemcpy(dst, data, n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Returns a pointer offset into the buffer. No allocation — just pointer math.
    float* at(size_t offset) const { return data + offset; }

    // Total bytes used.
    size_t bytes() const { return numel * sizeof(float); }
};

// GPU tensor for fp16 (__nv_bfloat16) data — same interface as Tensor but half the memory.
// Compute still happens in fp32 inside kernels; fp16 is just the storage format.
struct HalfTensor {
    __nv_bfloat16* data = nullptr;  // device (GPU) pointer (fp16 storage)
    size_t numel = 0;        // number of __nv_bfloat16 elements (NOT bytes)

    void allocate(size_t n) {
        free();
        numel = n;
        CUDA_CHECK(cudaMalloc(&data, n * sizeof(__nv_bfloat16)));
    }

    void free() {
        if (data) {
            CUDA_CHECK(cudaFree(data));
            data = nullptr;
            numel = 0;
        }
    }

    void zero() {
        if (data) {
            CUDA_CHECK(cudaMemset(data, 0, numel * sizeof(__nv_bfloat16)));
        }
    }

    __nv_bfloat16* at(size_t offset) const { return data + offset; }
    size_t bytes() const { return numel * sizeof(__nv_bfloat16); }
};

// One big GPU allocation that you carve into named sub-regions.
// Used for weights, gradients, and activations — allocate once, hand out pointers.
struct ParameterBlock {
    Tensor storage;    // the one big allocation
    size_t offset = 0; // cursor — how much has been carved out so far

    // Allocate total floats on GPU, zero them, reset cursor to start.
    void allocate(size_t total) {
        storage.allocate(total);
        storage.zero();
        offset = 0;
    }

    // Carve out n floats from the block. Returns a pointer to the start of the region.
    float* reserve(size_t n) {
        float* ptr = storage.at(offset);
        offset += n;
        return ptr;
    }

    // Free the entire block. All pointers from reserve() become invalid.
    void free() { storage.free(); }
};

// Same as ParameterBlock but for fp16 storage.
struct HalfParameterBlock {
    HalfTensor storage;
    size_t offset = 0;

    void allocate(size_t total) {
        storage.allocate(total);
        storage.zero();
        offset = 0;
    }

    __nv_bfloat16* reserve(size_t n) {
        __nv_bfloat16* ptr = storage.at(offset);
        offset += n;
        return ptr;
    }

    void free() { storage.free(); }
};
