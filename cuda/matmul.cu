#include "kernels.cuh"
// Forward: C[M,N] = A[M,K] @ B[K,N]

void matmul_forward(cublasHandle_t handle,
                    float* C, const float* A, const float* B,
                    int M, int N, int K){
    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N));
}
// Backward: dA[M,K] = dC[M,N] @ B^T[N,K]
//           dB[K,N] = A^T[K,M] @ dC[M,N]
void matmul_backward(cublasHandle_t handle,
                    float* dA, float* dB,
                    const float* dC, const float* A, const float* B,
                    int M, int N, int K){
    float alpha = 1.0f;
    float beta = 0.0f;          // dA: overwrite (fresh activation grad each micro-batch)
    float beta_accum = 1.0f;    // dB: accumulate weight grads across micro-batches (grad accumulation)

    // dA = dC @ B^T
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,  // transpose B (first arg), don't transpose dC (second arg)
        K, M, N,                   // output dA is [M,K] → cuBLAS gets (cols=K, rows=M, shared=N)
        &alpha,
        B, N,                      // B is [K,N], leading dim = N. Transposed → [N,K]
        dC, N,                     // dC is [M,N], leading dim = N. Not transposed.
        &beta,
        dA, K));                   // dA is [M,K], leading dim = K
    
    // dB = A^T @ dC  (+= so accum micro-batches sum; buffer zeroed once per step)
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,  // don't transpose dC (first arg), transpose A (second arg)
        N, K, M,                   // output dB is [K,N] → cuBLAS gets (cols=N, rows=K, shared=M)
        &alpha,
        dC, N,                     // dC is [M,N], leading dim = N. Not transposed.
        A, K,                      // A is [M,K], leading dim = K. Transposed → [K,M]
        &beta_accum,
        dB, N));                   // dB is [K,N], leading dim = N   
}


// ======================== fp16 overloads (tensor cores) ========================
// Same math, but fp16 storage + fp32 tensor core accumulation.
// cublasGemmEx replaces cublasSgemm — activates tensor cores on SM >= 7.0.

void matmul_forward(cublasHandle_t handle,
                    __nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
                    int M, int N, int K){
    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}

void matmul_backward(cublasHandle_t handle,
                    __nv_bfloat16* dA, __nv_bfloat16* dB,
                    const __nv_bfloat16* dC, const __nv_bfloat16* A, const __nv_bfloat16* B,
                    int M, int N, int K){
    float alpha = 1.0f;
    float beta = 0.0f;          // dA: overwrite (fresh activation grad each micro-batch)
    float beta_accum = 1.0f;    // dB: accumulate weight grads across micro-batches (grad accumulation)

    // dA = dC @ B^T
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        K, M, N,
        &alpha,
        B, CUDA_R_16BF, N,
        dC, CUDA_R_16BF, N,
        &beta,
        dA, CUDA_R_16BF, K,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));

    // dB = A^T @ dC  (+= so accum micro-batches sum; buffer zeroed once per step)
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, K, M,
        &alpha,
        dC, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta_accum,
        dB, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}