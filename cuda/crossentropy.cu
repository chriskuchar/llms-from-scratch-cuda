#include "kernels.cuh"
#include <math.h>

// --- crossentropy.cu ---
// Loss function — measures how far off the model's predictions are.

// Forward:
// probs = softmax(logits)
// loss = -log(probs[target])
// Mean loss = (-1 / (B * T)) * sum of loss[bt] for all positions
//
// Backward:
// dlogits[v] = (probs[v] - one_hot[v]) / (B * T)

// One thread per (batch, position) pair. Computes softmax inline + loss.
__global__ void crossentropy_forward_kernel(float* losses,        // per-position loss [B*T]
                                            float* probs,         // softmax output [B*T, V], written in-place
                                            const float* logits,  // raw model output [B*T, V]
                                            const int* targets,   // correct token IDs [B*T]
                                            int V) {
    int bt = blockIdx.x * blockDim.x + threadIdx.x;

    const float* logits_row = logits + bt * V;
    float* probs_row = probs + bt * V;

    float max_val = -INFINITY;                                  // track max logit for numerical stability
    for (int v = 0; v < V; v++) {
        if (logits_row[v] > max_val) max_val = logits_row[v];
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < V; v++) {
        float e = expf(logits_row[v] - max_val);               // subtract max for stability
        probs_row[v] = e;
        sum_exp += e;
    }

    for (int v = 0; v < V; v++) {
        probs_row[v] /= sum_exp;
    }

    int target = targets[bt];
    losses[bt] = -logf(probs_row[target]);
}

// Backward (fused softmax + cross-entropy):
// dlogits[bt, v] = (probs[bt, v] - (1 if v == target else 0)) / (B * T)

// One thread per (bt, v) element.
__global__ void crossentropy_softmax_backward_kernel(float* dlogits,        // logit gradients [B*T, V]
                                                     const float* probs,    // softmax probs from forward [B*T, V]
                                                     const int* targets,    // correct token IDs [B*T]
                                                     float scale,           // 1/(B*T), matches mean reduction
                                                     int B_T,
                                                     int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_T * V) return;

    int bt = idx / V;
    int v  = idx % V;

    float indicator = (v == targets[bt]) ? 1.0f : 0.0f;
    dlogits[idx] = (probs[idx] - indicator) * scale;
}

void crossentropy_forward(float* losses,          // per-position loss [B*T]
                          const float* logits,    // raw logits [B*T, V]
                          const int* targets,     // target token IDs [B*T]
                          int B,
                          int T,
                          int V) {
    int B_T = B * T;
    int block = 256;
    int grid = ceil_div(B_T, block);
    crossentropy_forward_kernel<<<grid, block>>>(losses, (float*)logits, logits, targets, V);
}

void crossentropy_softmax_backward(float* dlogits,        // logit gradients [B*T, V]
                                   const float* probs,    // softmax probs [B*T, V]
                                   const int* targets,    // target token IDs [B*T]
                                   int B,
                                   int T,
                                   int V) {
    int B_T = B * T;
    float scale = 1.0f / (float)B_T;                            // mean reduction factor
    int N = B_T * V;
    int block = 256;
    int grid = ceil_div(N, block);
    crossentropy_softmax_backward_kernel<<<grid, block>>>(dlogits, probs, targets, scale, B_T, V);
}

// ======================== fp16 overloads ========================
// fp16 logits input, fp32 loss output (loss needs precision).
// Probs written in-place over logits as fp16.

__global__ void crossentropy_forward_kernel_bf16(float* losses, __nv_bfloat16* probs,
    const __nv_bfloat16* logits, const int* targets, int V) {
    int bt = blockIdx.x * blockDim.x + threadIdx.x;

    const __nv_bfloat16* logits_row = logits + bt * V;
    __nv_bfloat16* probs_row = probs + bt * V;

    float max_val = -INFINITY;
    for (int v = 0; v < V; v++) {
        float lv = __bfloat162float(logits_row[v]);
        if (lv > max_val) max_val = lv;
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < V; v++) {
        float e = expf(__bfloat162float(logits_row[v]) - max_val);
        probs_row[v] = __float2bfloat16(e);
        sum_exp += e;
    }

    for (int v = 0; v < V; v++) {
        probs_row[v] = __float2bfloat16(__bfloat162float(probs_row[v]) / sum_exp);
    }

    int target = targets[bt];
    losses[bt] = -logf(__bfloat162float(probs_row[target]));  // loss stays fp32
}

__global__ void crossentropy_softmax_backward_kernel_bf16(__nv_bfloat16* dlogits,
    const __nv_bfloat16* probs, const int* targets, float scale, int B_T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_T * V) return;
    int bt = idx / V;
    int v  = idx % V;
    float indicator = (v == targets[bt]) ? 1.0f : 0.0f;
    dlogits[idx] = __float2bfloat16((__bfloat162float(probs[idx]) - indicator) * scale);
}

void crossentropy_forward(float* losses, const __nv_bfloat16* logits, const int* targets,
                          int B, int T, int V) {
    int B_T = B * T;
    int block = 256;
    int grid = ceil_div(B_T, block);
    crossentropy_forward_kernel_bf16<<<grid, block>>>(losses, (__nv_bfloat16*)logits, logits, targets, V);
}

void crossentropy_softmax_backward(__nv_bfloat16* dlogits, const __nv_bfloat16* probs,
                                   const int* targets, int B, int T, int V,
                                   float loss_scale) {
    int B_T = B * T;
    float scale = loss_scale / (float)B_T;  // bake loss_scale into the mean reduction
    int N = B_T * V;
    int block = 256;
    int grid = ceil_div(N, block);
    crossentropy_softmax_backward_kernel_bf16<<<grid, block>>>(dlogits, probs, targets, scale, B_T, V);
}
