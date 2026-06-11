#include "kernels.cuh"

// --- adamw.cu ---
// AdamW optimizer, one update per parameter per training step:
//   m = beta1 * m + (1 - beta1) * grad              (first moment)
//   v = beta2 * v + (1 - beta2) * grad^2            (second moment)
//   m_hat = m / (1 - beta1^t)                        (bias correction)
//   v_hat = v / (1 - beta2^t)                        (bias correction)
//   param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
// Weight decay is decoupled (applied to the weights directly), which is what distinguishes AdamW from Adam.

__global__ void adamw_update_kernel(float* params,        // model weights [N], updated in-place
                                    const float* grads,   // gradients [N]
                                    float* m,             // first moment buffer [N]
                                    float* v,             // second moment buffer [N]
                                    float lr,
                                    float beta1,
                                    float beta2,
                                    float eps,
                                    float weight_decay,
                                    int t,                // current training step (1-indexed, for bias correction)
                                    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float grad = grads[idx];
    float param = params[idx];

    float mi = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vi = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

    m[idx] = mi;
    v[idx] = vi;

    float m_hat = mi / (1.0f - powf(beta1, (float)t));
    float v_hat = vi / (1.0f - powf(beta2, (float)t));

    params[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
}

void adamw_update(float* params,        // model weights [N]
                  const float* grads,   // gradients [N]
                  float* m,             // first moment [N]
                  float* v,             // second moment [N]
                  float lr,
                  float beta1,
                  float beta2,
                  float eps,
                  float weight_decay,
                  int t,
                  int N) {
    int block = 256;
    int grid = ceil_div(N, block);
    adamw_update_kernel<<<grid, block>>>(params, grads, m, v,
                                         lr, beta1, beta2,
                                         eps, weight_decay, t, N);
}

// ======================== fp16 mixed precision AdamW ========================
// Reads fp16 gradients, updates fp32 master weights + m + v, writes fp16 model weights.
// Gradients were scaled by loss_scale during backward to avoid fp16 underflow; grad_scale (1/loss_scale) undoes it here.

__global__ void adamw_update_kernel_bf16(
    __nv_bfloat16* params,           // fp16 model weights (used in forward/backward)
    float* master_params,     // fp32 master copy (actual optimization target)
    const __nv_bfloat16* grads,      // fp16 gradients (scaled by loss_scale)
    float* m, float* v,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, float grad_scale, int t, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float grad = __bfloat162float(grads[idx]) * grad_scale;  // unscale in fp32
    float param = master_params[idx];                     // use fp32 master

    float mi = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vi = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = mi;
    v[idx] = vi;

    float m_hat = mi / (1.0f - powf(beta1, (float)t));
    float v_hat = vi / (1.0f - powf(beta2, (float)t));

    param -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);

    master_params[idx] = param;                   // update fp32 master
    params[idx] = __float2bfloat16(param);            // sync fp16 copy
}

void adamw_update(__nv_bfloat16* params, float* master_params, const __nv_bfloat16* grads,
                  float* m, float* v,
                  float lr, float beta1, float beta2, float eps,
                  float weight_decay, float grad_scale, int t, int N) {
    int block = 256;
    int grid = ceil_div(N, block);
    adamw_update_kernel_bf16<<<grid, block>>>(params, master_params, grads, m, v,
                                               lr, beta1, beta2, eps,
                                               weight_decay, grad_scale, t, N);
}

// GPU-side gradient norm: sum of squares reduced with atomicAdd
// fp32 version (for embedding grads)
__global__ void grad_norm_kernel_fp32(const float* grads, float* out, float scale, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < N) {
        float g = grads[idx] * scale;
        val = g * g;
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// fp16 version: also sanitizes inf/nan to zero in-place
__global__ void grad_norm_sanitize_kernel_bf16(__nv_bfloat16* grads, float* out, float scale, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < N) {
        float g = __bfloat162float(grads[idx]) * scale;
        if (g != g || fabsf(g) > 65000.0f) {
            grads[idx] = __float2bfloat16(0.0f);
        } else {
            val = g * g;
        }
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// Scale fp32 buffer in-place on GPU
__global__ void scale_kernel(float* data, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= scale;
}

void scale_grads(float* data, float scale, int N) {
    int block = 256;
    int grid = ceil_div(N, block);
    scale_kernel<<<grid, block>>>(data, scale, N);
}

// Convert fp32 → fp16 on GPU
__global__ void fp32_to_bf16_kernel(const float* src, __nv_bfloat16* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = __float2bfloat16(src[idx]);
}

void fp32_to_bf16(const float* src, __nv_bfloat16* dst, int N) {
    int block = 256;
    int grid = ceil_div(N, block);
    fp32_to_bf16_kernel<<<grid, block>>>(src, dst, N);
}

float compute_grad_norm_bf16(float* dwte_fp32, int emb_N,
                             __nv_bfloat16* grads_bf16, int rest_N, float loss_scale) {
    float scale = 1.0f / loss_scale;
    float* d_norm;
    CUDA_CHECK(cudaMalloc(&d_norm, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_norm, 0, sizeof(float)));

    int block = 256;
    // fp32 embedding grads
    int grid1 = ceil_div(emb_N, block);
    grad_norm_kernel_fp32<<<grid1, block>>>(dwte_fp32, d_norm, scale, emb_N);
    // fp16 rest (sanitizes inf/nan in-place)
    int grid2 = ceil_div(rest_N, block);
    grad_norm_sanitize_kernel_bf16<<<grid2, block>>>(grads_bf16, d_norm, scale, rest_N);

    float h_norm;
    CUDA_CHECK(cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_norm));
    return sqrtf(h_norm);
}
