#include "kernels.cuh"

// Forward:
// out = (x / sqrt(mean(x^2) + eps)) * weight
//
// Backward:
// dinp = rrms * (dout * weight - x * rrms^2 * sum(dout * weight * x) / C)
// dweight += dout * x * rrms

__global__ void rmsnorm_forward_kernel(float* out,          // normalized output [B*T, C]
    float* rrms,                                            // saved 1/rms per row [B*T], for backward
    const float* inp,                                       // input tensor [B*T, C]
    const float* weight,                                    // learnable scale [C]
    float eps,                                              // small constant to prevent div-by-zero (1e-5)
    int C){
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * C;

    float local_sum_sq = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        local_sum_sq += inp[base+j]*inp[base+j];
    }
    sdata[tid] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>=1){
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    float ssq = sdata[0] / (float)C;
    float inv_ssq = rsqrtf(ssq + eps);
    if (tid == 0) rrms[row] = inv_ssq;                                  // cache 1/rms for backward
    for (int j = tid; j < C; j+=blockDim.x){
        out[base + j] = inp[base+j]*inv_ssq * weight[j];
    }
}
// Backward:
// sum_val = sum(dout[c] * weight[c] * x[c])  over all c in row
// dinp[c] = rrms * (dout[c] * weight[c] - x[c] * rrms^2 * sum_val / C)
// dweight[c] += dout[c] * x[c] * rrms

__global__ void rmsnorm_backward_kernel(float* dinp,        // input gradient [B*T, C]
    float* dweight,                                         // weight gradient [C], accumulated
    const float* dout,                                      // upstream gradient [B*T, C]
    const float* inp,                                       // saved input from forward [B*T, C]
    const float* weight,                                    // learnable scale [C]
    const float* rrms,                                      // saved 1/rms from forward [B*T]
    int C){
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * C;

    float local_sum_val = 0.0f;
    for(int j = tid; j < C; j += blockDim.x){
        local_sum_val += dout[base+j]*weight[j]*inp[base+j];
    }
    sdata[tid] = local_sum_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s >0 ; s>>=1){
        if(tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    float sum_val = sdata[0];
    float rms_inv = rrms[row];

    for(int j = tid; j<C; j+=blockDim.x){
        float w_dy = dout[base+j] * weight[j];
        dinp[base + j] = rms_inv * (w_dy - inp[base+j] * rms_inv * rms_inv * sum_val / C);
        atomicAdd(&dweight[j], dout[base+j] * inp[base+j] * rms_inv);
    }
}
// --- rmsnorm.cu ---
// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
void rmsnorm_forward(float* out,            // normalized output [B*T, C]
    float* rrms,                            // saved 1/rms per row [B*T]
    const float* inp,                       // input tensor [B*T, C]
    const float* weight,                    // learnable scale [C]
    float eps,                              // stability constant (1e-5)
    int B_T,
    int C){
    int block = 256;
    rmsnorm_forward_kernel<<<B_T, block>>>(out, rrms, inp, weight, eps, C);  // one block per row
}


// Backward through RMSNorm. Given dout, compute:
//   dweight[c] += sum over rows of: dout[row][c] * inp[row][c] * rrms[row]
//   dinp[row][c] = weight[c] * rrms * (dout[row][c] - inp[row][c] * rrms^2 * (1/C) * sum(dout * weight * inp))    
void rmsnorm_backward(float* dinp,          // input gradient [B*T, C]
    float* dweight,                         // weight gradient [C], accumulated
    const float* dout,                      // upstream gradient [B*T, C]
    const float* inp,                       // saved input from forward [B*T, C]
    const float* weight,                    // learnable scale [C]
    const float* rrms,                      // saved 1/rms from forward [B*T]
    int B_T,
    int C){
    int block = 256;
    rmsnorm_backward_kernel<<<B_T, block>>>(dinp, dweight, dout, inp, weight, rrms, C);  // one block per row
}

// ======================== fp16 overloads ========================
// Same math but fp16 I/O. Internal accumulation stays fp32 for precision.
// rrms stays float* (one scalar per row, needs precision).

__global__ void rmsnorm_forward_kernel_bf16(__nv_bfloat16* out, float* rrms,
    const __nv_bfloat16* inp, const __nv_bfloat16* weight, float eps, int C) {
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * C;

    float local_sum_sq = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        float x = __bfloat162float(inp[base+j]);
        local_sum_sq += x * x;
    }
    sdata[tid] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    float ssq = sdata[0] / (float)C;
    float inv_ssq = rsqrtf(ssq + eps);
    if (tid == 0) rrms[row] = inv_ssq;
    for (int j = tid; j < C; j += blockDim.x) {
        float x = __bfloat162float(inp[base+j]);
        float w = __bfloat162float(weight[j]);
        out[base + j] = __float2bfloat16(x * inv_ssq * w);
    }
}

__global__ void rmsnorm_backward_kernel_bf16(__nv_bfloat16* dinp, __nv_bfloat16* dweight,
    const __nv_bfloat16* dout, const __nv_bfloat16* inp, const __nv_bfloat16* weight,
    const float* rrms, int C) {
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * C;

    float local_sum_val = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        local_sum_val += __bfloat162float(dout[base+j]) * __bfloat162float(weight[j]) * __bfloat162float(inp[base+j]);
    }
    sdata[tid] = local_sum_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    float sum_val = sdata[0];
    float rms_inv = rrms[row];

    for (int j = tid; j < C; j += blockDim.x) {
        float dy = __bfloat162float(dout[base+j]);
        float w  = __bfloat162float(weight[j]);
        float x  = __bfloat162float(inp[base+j]);
        float w_dy = dy * w;
        dinp[base + j] = __float2bfloat16(rms_inv * (w_dy - x * rms_inv * rms_inv * sum_val / C));
        atomicAddBf16(&dweight[j], dy * x * rms_inv);
    }
}

void rmsnorm_forward(__nv_bfloat16* out, float* rrms, const __nv_bfloat16* inp,
    const __nv_bfloat16* weight, float eps, int B_T, int C) {
    int block = 256;
    rmsnorm_forward_kernel_bf16<<<B_T, block>>>(out, rrms, inp, weight, eps, C);
}

void rmsnorm_backward(__nv_bfloat16* dinp, __nv_bfloat16* dweight, const __nv_bfloat16* dout,
    const __nv_bfloat16* inp, const __nv_bfloat16* weight, const float* rrms, int B_T, int C) {
    int block = 256;
    rmsnorm_backward_kernel_bf16<<<B_T, block>>>(dinp, dweight, dout, inp, weight, rrms, C);
}
