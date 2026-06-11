#include "kernels.cuh"

// --- softmax.cu ---
// Forward:
// out[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
//
// Backward (fused into attention/crossentropy, not here):
// dx[i] = out[i] * (dout[i] - sum(dout * out))

__global__ void softmax_forward_kernel(float* output, const float* input, int rows, int cols) {
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * cols;

    float local_max = -INFINITY;
    for(int j = tid; j < cols; j += blockDim.x){
        local_max = fmaxf(local_max, input[base+j]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Step 1: Row max via reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Step 2: exp(x - max) and sum via reduction
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float e = expf(input[base + j] - row_max);
        output[row * cols + j] = e;   // store exp values for now, divide later
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // Step 3: Normalize
    for (int j = tid; j < cols; j += blockDim.x) {
        output[base + j] /= row_sum;
    }
}

void softmax_forward(float* output, const float* input, int rows, int cols){
    int block = 256;
    softmax_forward_kernel<<<rows, block>>>(output, input, rows, cols);
}

// ======================== fp16 overload ========================
// fp16 I/O, fp32 accumulation in shared memory for numerical stability.

__global__ void softmax_forward_kernel_bf16(__nv_bfloat16* output, const __nv_bfloat16* input, int rows, int cols) {
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * cols;

    float local_max = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) {
        local_max = fmaxf(local_max, __bfloat162float(input[base+j]));
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float e = expf(__bfloat162float(input[base + j]) - row_max);
        output[base + j] = __float2bfloat16(e);
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int j = tid; j < cols; j += blockDim.x) {
        output[base + j] = __float2bfloat16(__bfloat162float(output[base + j]) / row_sum);
    }
}

void softmax_forward(__nv_bfloat16* output, const __nv_bfloat16* input, int rows, int cols) {
    int block = 256;
    softmax_forward_kernel_bf16<<<rows, block>>>(output, input, rows, cols);
}