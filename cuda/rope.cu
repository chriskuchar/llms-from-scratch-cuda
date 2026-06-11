#include "kernels.cuh"

// --- rope.cu ---
// RoPE = Rotary Position Embedding

// Forward:
// freq = 1 / (theta ^ (2*i / head_dim))
// angle = t * freq
// q[2i]_new   = q[2i]*cos(angle) - q[2i+1]*sin(angle)
// q[2i+1]_new = q[2i]*sin(angle) + q[2i+1]*cos(angle)

// Same rotation applied to K. Both modified in-place.
// One thread per dimension pair, for one head at one position.
// Handles both Q (n_head heads) and K (n_kv_head heads) in one launch.
__global__ void rope_forward_kernel(float* q,          // query tensor [B*T, n_head * head_dim], modified in-place
                                    float* k,          // key tensor [B*T, n_kv_head * head_dim], modified in-place
                                    int T,
                                    int n_head,        // number of query heads
                                    int n_kv_head,     // number of KV heads
                                    int head_dim,      // dimensions per head
                                    float theta) {     // RoPE frequency base
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = n_head + n_kv_head;
    int half_hd = head_dim / 2;                                 // rotation pairs per head

    int i  = idx % half_hd;                                     // which pair within this head
    int h  = (idx / half_hd) % total_heads;                     // which head (Q first, then K)
    int bt = idx / (half_hd * total_heads);                     // which (batch, position)
    int t  = bt % T;                                            // sequence position → determines angle

    float freq = 1.0f / powf(theta, (2.0f * i) / (float)head_dim);
    float angle = t * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float* target;
    int head_in_target;
    if (h < n_head) {                                           // first n_head indices → Q
        target = q;
        head_in_target = h;
    } else {                                                    // remaining indices → K
        target = k;
        head_in_target = h - n_head;
    }

    int num_heads_in_target = (h < n_head) ? n_head : n_kv_head;
    int base = bt * (num_heads_in_target * head_dim) + head_in_target * head_dim + 2 * i;  // flat offset for this pair

    float x0 = target[base];
    float x1 = target[base + 1];

    target[base]     = x0 * cos_val - x1 * sin_val;
    target[base + 1] = x0 * sin_val + x1 * cos_val;
}

// Backward (inverse rotation — negate the angle):
// dq[2i]   =  dq_out[2i]*cos(angle) + dq_out[2i+1]*sin(angle)
// dq[2i+1] = -dq_out[2i]*sin(angle) + dq_out[2i+1]*cos(angle)
// Same for dk. Rotation is orthogonal so inverse = negate angle.
__global__ void rope_backward_kernel(float* dq,         // query gradient [B*T, n_head * head_dim], modified in-place
                                     float* dk,         // key gradient [B*T, n_kv_head * head_dim], modified in-place
                                     int T,
                                     int n_head,        // number of query heads
                                     int n_kv_head,     // number of KV heads
                                     int head_dim,      // dimensions per head
                                     float theta) {     // RoPE frequency base
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = n_head + n_kv_head;
    int half_hd = head_dim / 2;                                 // rotation pairs per head

    int i  = idx % half_hd;                                     // which pair within this head
    int h  = (idx / half_hd) % total_heads;                     // which head (Q first, then K)
    int bt = idx / (half_hd * total_heads);                     // which (batch, position)
    int t  = bt % T;                                            // sequence position → determines angle

    float freq = 1.0f / powf(theta, (2.0f * i) / (float)head_dim);
    float angle = t * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float* target;
    int head_in_target;
    if (h < n_head) {                                           // first n_head indices → dQ
        target = dq;
        head_in_target = h;
    } else {                                                    // remaining indices → dK
        target = dk;
        head_in_target = h - n_head;
    }

    int num_heads_in_target = (h < n_head) ? n_head : n_kv_head;
    int base = bt * (num_heads_in_target * head_dim) + head_in_target * head_dim + 2 * i;  // flat offset for this pair

    float dx0 = target[base];
    float dx1 = target[base + 1];

    target[base]     =  dx0 * cos_val + dx1 * sin_val;
    target[base + 1] = -dx0 * sin_val + dx1 * cos_val;
}

void rope_forward(float* q,          // query tensor [B*T, C], modified in-place
                  float* k,          // key tensor [B*T, kv_dim], modified in-place
                  int B,
                  int T,
                  int n_head,        // query heads
                  int n_kv_head,     // KV heads
                  int head_dim,      // dims per head
                  float theta) {     // RoPE base
    int N = B * T * (n_head + n_kv_head) * (head_dim / 2);      // total rotation pairs across batch
    int block = 256;
    int grid = ceil_div(N, block);
    rope_forward_kernel<<<grid, block>>>(q, k, T, n_head, n_kv_head, head_dim, theta);
}

void rope_backward(float* dq,         // query gradient [B*T, C], modified in-place
                   float* dk,         // key gradient [B*T, kv_dim], modified in-place
                   int B,
                   int T,
                   int n_head,        // query heads
                   int n_kv_head,     // KV heads
                   int head_dim,      // dims per head
                   float theta) {     // RoPE base
    int N = B * T * (n_head + n_kv_head) * (head_dim / 2);      // total rotation pairs across batch
    int block = 256;
    int grid = ceil_div(N, block);
    rope_backward_kernel<<<grid, block>>>(dq, dk, T, n_head, n_kv_head, head_dim, theta);
}

// ======================== fp16 overloads ========================
// Same rotation math but fp16 I/O. Trig computed in fp32 for precision.

__global__ void rope_forward_kernel_bf16(__nv_bfloat16* q, __nv_bfloat16* k,
    int T, int n_head, int n_kv_head, int head_dim, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = n_head + n_kv_head;
    int half_hd = head_dim / 2;

    int i  = idx % half_hd;
    int h  = (idx / half_hd) % total_heads;
    int bt = idx / (half_hd * total_heads);
    int t  = bt % T;

    float freq = 1.0f / powf(theta, (2.0f * i) / (float)head_dim);
    float angle = t * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    __nv_bfloat16* target;
    int head_in_target;
    if (h < n_head) { target = q; head_in_target = h; }
    else { target = k; head_in_target = h - n_head; }

    int num_heads_in_target = (h < n_head) ? n_head : n_kv_head;
    int base = bt * (num_heads_in_target * head_dim) + head_in_target * head_dim + 2 * i;

    float x0 = __bfloat162float(target[base]);
    float x1 = __bfloat162float(target[base + 1]);

    target[base]     = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    target[base + 1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

__global__ void rope_backward_kernel_bf16(__nv_bfloat16* dq, __nv_bfloat16* dk,
    int T, int n_head, int n_kv_head, int head_dim, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = n_head + n_kv_head;
    int half_hd = head_dim / 2;

    int i  = idx % half_hd;
    int h  = (idx / half_hd) % total_heads;
    int bt = idx / (half_hd * total_heads);
    int t  = bt % T;

    float freq = 1.0f / powf(theta, (2.0f * i) / (float)head_dim);
    float angle = t * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    __nv_bfloat16* target;
    int head_in_target;
    if (h < n_head) { target = dq; head_in_target = h; }
    else { target = dk; head_in_target = h - n_head; }

    int num_heads_in_target = (h < n_head) ? n_head : n_kv_head;
    int base = bt * (num_heads_in_target * head_dim) + head_in_target * head_dim + 2 * i;

    float dx0 = __bfloat162float(target[base]);
    float dx1 = __bfloat162float(target[base + 1]);

    target[base]     = __float2bfloat16( dx0 * cos_val + dx1 * sin_val);
    target[base + 1] = __float2bfloat16(-dx0 * sin_val + dx1 * cos_val);
}

void rope_forward(__nv_bfloat16* q, __nv_bfloat16* k, int B, int T,
                  int n_head, int n_kv_head, int head_dim, float theta) {
    int N = B * T * (n_head + n_kv_head) * (head_dim / 2);
    int block = 256;
    int grid = ceil_div(N, block);
    rope_forward_kernel_bf16<<<grid, block>>>(q, k, T, n_head, n_kv_head, head_dim, theta);
}

void rope_backward(__nv_bfloat16* dq, __nv_bfloat16* dk, int B, int T,
                   int n_head, int n_kv_head, int head_dim, float theta) {
    int N = B * T * (n_head + n_kv_head) * (head_dim / 2);
    int block = 256;
    int grid = ceil_div(N, block);
    rope_backward_kernel_bf16<<<grid, block>>>(dq, dk, T, n_head, n_kv_head, head_dim, theta);
}
