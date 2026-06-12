#include "model.h"
#include "kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// BUILD — allocate all GPU memory, carve pointers, init weights.
void Model::build(ModelConfig cfg, int B, int T) {
    config = cfg;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    int C  = config.n_embd;
    int V  = config.vocab_size;
    int L  = config.n_layer;
    int kv = config.kv_dim();      // total KV dim across heads
    int ff = config.ffn_hidden;
    int nh = config.n_head;
    int BT = B * T;                // positions per batch

    // Count params. Term order must match the reserve() order below.
    size_t n_params = (size_t)V * C       // wte    [V × C]
                    + (size_t)V * C       // w_out  [V × C]
                    + C                   // rmsf_w [C]
                    + (size_t)L * (       // per layer:
                        C                 //   rms1_w [C]
                        + (size_t)C * C   //   wq     [C × C]
                        + (size_t)C * kv  //   wk     [C × kv]
                        + (size_t)C * kv  //   wv     [C × kv]
                        + (size_t)C * C   //   wo     [C × C]
                        + C               //   rms2_w [C]
                        + (size_t)C * ff  //   w_gate [C × ff]
                        + (size_t)C * ff  //   w_up   [C × ff]
                        + (size_t)ff * C  //   w_down [ff × C]
                    );

    // One flat allocation for all weights, one for all gradients (same layout).
    params.allocate(n_params);
    grads.allocate(n_params);

    // Carve weight pointers from the flat buffer. Order must match the count above.
    wte    = params.reserve(V * C);
    w_out  = params.reserve(V * C);
    rmsf_w = params.reserve(C);

    // CPU arrays of L GPU pointers, one per layer.
    rms1_w = new float*[L];
    wq     = new float*[L];
    wk     = new float*[L];
    wv     = new float*[L];
    wo     = new float*[L];
    rms2_w = new float*[L];
    w_gate = new float*[L];
    w_up   = new float*[L];
    w_down = new float*[L];

    for (int l = 0; l < L; l++) {
        rms1_w[l] = params.reserve(C);
        wq[l]     = params.reserve(C * C);
        wk[l]     = params.reserve(C * kv);
        wv[l]     = params.reserve(C * kv);
        wo[l]     = params.reserve(C * C);
        rms2_w[l] = params.reserve(C);
        w_gate[l] = params.reserve(C * ff);
        w_up[l]   = params.reserve(C * ff);
        w_down[l] = params.reserve(ff * C);
    }

    // Gradient pointers: same order and sizes as the weights.
    dwte    = grads.reserve(V * C);
    dw_out  = grads.reserve(V * C);
    drmsf_w = grads.reserve(C);

    drms1_w = new float*[L];
    dwq     = new float*[L];
    dwk     = new float*[L];
    dwv     = new float*[L];
    dwo     = new float*[L];
    drms2_w = new float*[L];
    dw_gate = new float*[L];
    dw_up   = new float*[L];
    dw_down = new float*[L];

    for (int l = 0; l < L; l++) {
        drms1_w[l] = grads.reserve(C);
        dwq[l]     = grads.reserve(C * C);
        dwk[l]     = grads.reserve(C * kv);
        dwv[l]     = grads.reserve(C * kv);
        dwo[l]     = grads.reserve(C * C);
        drms2_w[l] = grads.reserve(C);
        dw_gate[l] = grads.reserve(C * ff);
        dw_up[l]   = grads.reserve(C * ff);
        dw_down[l] = grads.reserve(ff * C);
    }

    // Activations: forward intermediates plus backward gradient buffers,
    // all carved from one allocation.
    size_t n_acts = (size_t)BT * C            // x          [BT × C]
                  + (size_t)BT * C            // ln1_out    [BT × C]
                  + BT                        // rrms1      [BT]
                  + (size_t)BT * C            // q          [BT × C]
                  + (size_t)BT * kv           // k          [BT × kv]
                  + (size_t)BT * kv           // v          [BT × kv]
                  + (size_t)B * nh * T * T    // att        [B × nh × T × T]
                  + (size_t)BT * C            // attn_out   [BT × C]
                  + (size_t)BT * C            // ln2_out    [BT × C]
                  + BT                        // rrms2      [BT]
                  + (size_t)BT * ff           // gate_buf   [BT × ff]
                  + (size_t)BT * ff           // up_buf     [BT × ff]
                  + (size_t)BT * ff           // hidden_buf [BT × ff]
                  + (size_t)BT * C            // mlp_out    [BT × C]
                  + (size_t)BT * C            // ln_final   [BT × C]
                  + BT                        // rrms_final [BT]
                  + (size_t)BT * V            // logits     [BT × V]
                  + (size_t)BT * V            // probs      [BT × V]
                  + BT                        // losses     [BT]
                  + (size_t)BT * C            // dx         [BT × C]
                  + (size_t)BT * C            // dln1_out   [BT × C]
                  + (size_t)BT * C            // dln2_out   [BT × C]
                  + (size_t)BT * C            // dattn_out  [BT × C]
                  + (size_t)BT * C            // dmlp_out   [BT × C]
                  + (size_t)BT * C            // dln_final  [BT × C]
                  + (size_t)BT * V            // dlogits    [BT × V]
                  + (size_t)BT * C;           // dx_tmp     [BT × C]

    acts.allocate(n_acts);

    x          = acts.reserve(BT * C);
    ln1_out    = acts.reserve(BT * C);
    rrms1      = acts.reserve(BT);
    q          = acts.reserve(BT * C);
    k          = acts.reserve(BT * kv);
    v          = acts.reserve(BT * kv);
    att        = acts.reserve(B * nh * T * T);
    attn_out   = acts.reserve(BT * C);
    ln2_out    = acts.reserve(BT * C);
    rrms2      = acts.reserve(BT);
    gate_buf   = acts.reserve(BT * ff);
    up_buf     = acts.reserve(BT * ff);
    hidden_buf = acts.reserve(BT * ff);
    mlp_out    = acts.reserve(BT * C);
    ln_final   = acts.reserve(BT * C);
    rrms_final = acts.reserve(BT);
    logits     = acts.reserve(BT * V);
    probs      = acts.reserve(BT * V);
    losses     = acts.reserve(BT);
    dx         = acts.reserve(BT * C);
    dln1_out   = acts.reserve(BT * C);
    dln2_out   = acts.reserve(BT * C);
    dattn_out  = acts.reserve(BT * C);
    dmlp_out   = acts.reserve(BT * C);
    dln_final  = acts.reserve(BT * C);
    dlogits    = acts.reserve(BT * V);
    dx_tmp     = acts.reserve(BT * C);

    // Gradient-checkpointing buffers: x_ckpt holds each layer's residual-stream input
    // (saved in forward); backward recomputes per-layer activations from these.
    CUDA_CHECK(cudaMalloc(&x_ckpt, (size_t)L * BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&x_mid, (size_t)BT * C * sizeof(float)));

    adam_m.allocate(n_params);
    adam_v.allocate(n_params);

    // Init: N(0, 0.02) via Box-Muller on CPU, then RMSNorm weights set to 1.0.
    size_t total = params.storage.numel;
    float* cpu_buf = (float*)malloc(total * sizeof(float));
    srand(42);
    for (size_t i = 0; i < total; i += 2) {
        float u1 = ((float)rand() / RAND_MAX);
        float u2 = ((float)rand() / RAND_MAX);
        if (u1 < 1e-7f) u1 = 1e-7f;          // avoid log(0)
        float mag = 0.02f * sqrtf(-2.0f * logf(u1));         // stddev = 0.02
        cpu_buf[i]     = mag * cosf(2.0f * 3.14159265f * u2);
        if (i + 1 < total) {
            cpu_buf[i + 1] = mag * sinf(2.0f * 3.14159265f * u2);
        }
    }
    CUDA_CHECK(cudaMemcpy(params.storage.data, cpu_buf,
                          total * sizeof(float), cudaMemcpyHostToDevice));
    ::free(cpu_buf);

    // RMSNorm weights start at 1.0 (scale only).
    float* ones = (float*)malloc(C * sizeof(float));
    for (int i = 0; i < C; i++) ones[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(rmsf_w, ones, C * sizeof(float), cudaMemcpyHostToDevice));
    for (int l = 0; l < L; l++) {
        CUDA_CHECK(cudaMemcpy(rms1_w[l], ones, C * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(rms2_w[l], ones, C * sizeof(float), cudaMemcpyHostToDevice));
    }
    ::free(ones);

    printf("Model: %d layers, %d heads (%d kv), %d embd, %d ffn\n",
           L, nh, config.n_kv_head, C, ff);
    printf("Parameters: %.1fM\n", (float)n_params / 1e6f);
    printf("Activation memory: %.1fMB\n", (float)(n_acts * sizeof(float)) / (1024.0f * 1024.0f));
}

// LOAD — restore weights from a checkpoint file.
// Inverse of the checkpoint save in main.cpp: raw float32s in params.storage order.
void Model::load(const char* checkpoint_path) {
    size_t total = params.storage.numel;

    FILE* f = fopen(checkpoint_path, "rb");
    if (!f) { printf("Error: could not open checkpoint %s\n", checkpoint_path); exit(1); }

    float* cpu_buf = (float*)malloc(total * sizeof(float));
    size_t read = fread(cpu_buf, sizeof(float), total, f);
    if (read != total) {
        printf("Error: checkpoint has %zu floats, expected %zu\n", read, total);
        exit(1);
    }
    fclose(f);

    CUDA_CHECK(cudaMemcpy(params.storage.data, cpu_buf,
                          total * sizeof(float), cudaMemcpyHostToDevice));
    ::free(cpu_buf);
    printf("Loaded checkpoint: %s (%zu params)\n", checkpoint_path, total);
}

// FORWARD — run input tokens through the entire model + loss.
void Model::forward(const int* input_ids, const int* targets, int B, int T) {
    int C   = config.n_embd;
    int L   = config.n_layer;
    int nh  = config.n_head;
    int nkv = config.n_kv_head;
    int ff  = config.ffn_hidden;
    int V   = config.vocab_size;
    int BT  = B * T;

    embedding_forward(x, wte, input_ids, B, T, C);

    for (int l = 0; l < L; l++) {

        // Checkpoint this layer's input so backward can recompute its forward.
        CUDA_CHECK(cudaMemcpy(x_ckpt + (size_t)l * BT * C, x,
                              (size_t)BT * C * sizeof(float), cudaMemcpyDeviceToDevice));

        rmsnorm_forward(ln1_out, rrms1, x, rms1_w[l], config.norm_eps, BT, C);


        #ifdef USE_FLASH_ATTN
            // Flash path: scores → mask → softmax → context fused into one tiled
            // kernel, saving lse instead of the full T×T att matrix for backward.
            attention_forward(attn_out, q, k, v, lse, 
                ln1_out, wq[l], wk[l], wv[l], wo[l],
                B, T, C, nh, nkv, config.rope_theta, cublas_handle);
        #else
            // Grouped-query attention: Q/K/V → RoPE → scores → causal mask → softmax → context → W_o.
            attention_forward(attn_out, q, k, v, att,
                ln1_out, wq[l], wk[l], wv[l], wo[l],
                B, T, C, nh, nkv, config.rope_theta, cublas_handle);
        #endif
        residual_forward(x, x, attn_out, BT * C);

        rmsnorm_forward(ln2_out, rrms2, x, rms2_w[l], config.norm_eps, BT, C);

        swiglu_forward(mlp_out, gate_buf, up_buf, hidden_buf,
                       ln2_out, w_gate[l], w_up[l], w_down[l],
                       BT, C, ff, cublas_handle);

        residual_forward(x, x, mlp_out, BT * C);
    }

    rmsnorm_forward(ln_final, rrms_final, x, rmsf_w, config.norm_eps, BT, C);

    matmul_forward(cublas_handle, logits, ln_final, w_out, BT, V, C);

    crossentropy_forward(losses, logits, targets, B, T, V);
    probs = logits;            // crossentropy writes probs in-place over logits
    saved_targets = targets;   // save for backward()

    float* cpu_losses = (float*)malloc(BT * sizeof(float));
    CUDA_CHECK(cudaMemcpy(cpu_losses, losses, BT * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < BT; i++) sum += cpu_losses[i];
    loss = sum / (float)BT;
    ::free(cpu_losses);
}

// BACKWARD — reverse the forward pass, compute all gradients.
void Model::backward(int B, int T) {
    int C   = config.n_embd;
    int L   = config.n_layer;
    int nh  = config.n_head;
    int nkv = config.n_kv_head;
    int ff  = config.ffn_hidden;
    int V   = config.vocab_size;
    int BT  = B * T;

    crossentropy_softmax_backward(dlogits, probs, saved_targets, B, T, V);

    matmul_backward(cublas_handle, dln_final, dw_out, dlogits, ln_final, w_out, BT, V, C);

    rmsnorm_backward(dx, drmsf_w, dln_final, x, rmsf_w, rrms_final, BT, C);

    // The single activation buffers only hold the last layer's values, so for each layer
    // we recompute its forward from the checkpointed input x_ckpt[l], then run backward.
    for (int l = L - 1; l >= 0; l--) {
        float* x_in = x_ckpt + (size_t)l * BT * C;   // this layer's residual-stream input

        // Recompute layer l's forward activations.
        rmsnorm_forward(ln1_out, rrms1, x_in, rms1_w[l], config.norm_eps, BT, C);
        attention_forward(attn_out, q, k, v, att,
            ln1_out, wq[l], wk[l], wv[l], wo[l],
            B, T, C, nh, nkv, config.rope_theta, cublas_handle);
        residual_forward(x_mid, x_in, attn_out, BT * C);   // x_mid = input to rms2
        rmsnorm_forward(ln2_out, rrms2, x_mid, rms2_w[l], config.norm_eps, BT, C);
        swiglu_forward(mlp_out, gate_buf, up_buf, hidden_buf,
                       ln2_out, w_gate[l], w_up[l], w_down[l],
                       BT, C, ff, cublas_handle);

        // MLP residual: both branches get a copy of dx.
        CUDA_CHECK(cudaMemcpy(dx_tmp, dx, BT * C * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(dmlp_out, 0, BT * C * sizeof(float)));
        residual_backward(dx, dmlp_out, dx_tmp, BT * C);

        swiglu_backward(dln2_out, dw_gate[l], dw_up[l], dw_down[l],
                        dmlp_out, ln2_out, gate_buf, up_buf,
                        w_gate[l], w_up[l], w_down[l],
                        BT, C, ff, cublas_handle);

        rmsnorm_backward(dx_tmp, drms2_w[l], dln2_out, x_mid, rms2_w[l], rrms2, BT, C);  // input was x_mid
        residual_forward(dx, dx, dx_tmp, BT * C);   // dx += dx_tmp

        CUDA_CHECK(cudaMemcpy(dx_tmp, dx, BT * C * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(dattn_out, 0, BT * C * sizeof(float)));
        residual_backward(dx, dattn_out, dx_tmp, BT * C);
        #ifdef USE_FLASH_ATTN
            // Flash path recomputes scores from q,k and uses saved lse to
            // renormalize softmax, so no T×T att matrix is needed.
            attention_backward(dln1_out, dwq[l], dwk[l], dwv[l], dwo[l],
                dattn_out, ln1_out, q, k, v, lse,
                wq[l], wk[l], wv[l], wo[l],
                B, T, C, nh, nkv, config.rope_theta, cublas_handle);
        #else
            attention_backward(dln1_out, dwq[l], dwk[l], dwv[l], dwo[l],
                dattn_out, ln1_out, q, k, v, att,
                wq[l], wk[l], wv[l], wo[l],
                B, T, C, nh, nkv, config.rope_theta, cublas_handle);
        #endif
        rmsnorm_backward(dx_tmp, drms1_w[l], dln1_out, x_in, rms1_w[l], rrms1, BT, C);  // input was x_in
        residual_forward(dx, dx, dx_tmp, BT * C);   // dx += dx_tmp
    }

    // Embedding backward is skipped here: input_ids aren't stored in Model,
    // so the training loop calls embedding_backward(dwte, dx, input_ids, ...) directly.
}

// ZERO GRAD — clear all gradient buffers before each step.
void Model::zero_grad() {
    CUDA_CHECK(cudaMemset(grads.storage.data, 0, grads.storage.bytes()));
}

// UPDATE — run AdamW optimizer on all parameters.
void Model::update(float lr, TrainConfig tcfg, int step) {
    int N = (int)params.storage.numel;
    adamw_update(params.storage.data,
                 grads.storage.data,
                 adam_m.data,
                 adam_v.data,
                 lr,
                 tcfg.beta1,
                 tcfg.beta2,
                 tcfg.eps,
                 tcfg.weight_decay,
                 step,
                 N);
}

// FREE — release all GPU and CPU memory.
void Model::free() {
    cublasDestroy(cublas_handle);
    params.free();
    grads.free();
    acts.free();                            // free activation buffer
    CUDA_CHECK(cudaFree(x_ckpt));            // free gradient-checkpoint buffers
    CUDA_CHECK(cudaFree(x_mid));
    adam_m.free();                          // free first moment buffer
    adam_v.free();                          // free second moment buffer

    // free CPU arrays of per-layer GPU pointers
    delete[] rms1_w;  delete[] wq;     delete[] wk;
    delete[] wv;      delete[] wo;     delete[] rms2_w;
    delete[] w_gate;  delete[] w_up;   delete[] w_down;
    delete[] drms1_w; delete[] dwq;    delete[] dwk;
    delete[] dwv;     delete[] dwo;    delete[] drms2_w;
    delete[] dw_gate; delete[] dw_up;  delete[] dw_down;
}

// ======================== ModelBF16 =============================
// Mixed precision: bf16 storage, fp32 compute, fp32 optimizer.
// Enables tensor cores on SM >= 8.0 (RTX 30xx, 40xx, A100, etc.)

// Convert an fp32 CPU array to bf16 and upload to GPU.
static void upload_fp32_as_bf16(__nv_bfloat16* gpu_dst, const float* cpu_src, size_t n) {
    __nv_bfloat16* cpu_bf16 = new __nv_bfloat16[n];
    for (size_t i = 0; i < n; i++) cpu_bf16[i] = __float2bfloat16(cpu_src[i]);
    CUDA_CHECK(cudaMemcpy(gpu_dst, cpu_bf16, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    delete[] cpu_bf16;
}

void ModelBF16::build(ModelConfig cfg, int B, int T) {
    config = cfg;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));  // enable tensor cores

    int C  = config.n_embd;
    int V  = config.vocab_size;
    int L  = config.n_layer;
    int kv = config.kv_dim();
    int ff = config.ffn_hidden;
    int nh = config.n_head;
    int BT = B * T;

    size_t n_params = (size_t)V * C + (size_t)V * C + C
                    + (size_t)L * (C + (size_t)C*C + (size_t)C*kv + (size_t)C*kv
                                    + (size_t)C*C + C + (size_t)C*ff + (size_t)C*ff + (size_t)ff*C);

    // fp16 weights and gradients
    params.allocate(n_params);
    grads.allocate(n_params);

    // fp32 master weights + optimizer state
    master_params.allocate(n_params);
    adam_m.allocate(n_params);
    adam_v.allocate(n_params);

    // carve fp16 weight pointers
    wte    = params.reserve(V * C);
    w_out  = params.reserve(V * C);
    rmsf_w = params.reserve(C);

    rms1_w = new __nv_bfloat16*[L]; wq = new __nv_bfloat16*[L]; wk = new __nv_bfloat16*[L];
    wv = new __nv_bfloat16*[L]; wo = new __nv_bfloat16*[L]; rms2_w = new __nv_bfloat16*[L];
    w_gate = new __nv_bfloat16*[L]; w_up = new __nv_bfloat16*[L]; w_down = new __nv_bfloat16*[L];

    for (int l = 0; l < L; l++) {
        rms1_w[l] = params.reserve(C);
        wq[l]     = params.reserve(C * C);
        wk[l]     = params.reserve(C * kv);
        wv[l]     = params.reserve(C * kv);
        wo[l]     = params.reserve(C * C);
        rms2_w[l] = params.reserve(C);
        w_gate[l] = params.reserve(C * ff);
        w_up[l]   = params.reserve(C * ff);
        w_down[l] = params.reserve(ff * C);
    }

    // carve fp16 gradient pointers (same layout)
    dwte    = grads.reserve(V * C);  // placeholder in fp16 block (not used for embedding grads)
    dw_out  = grads.reserve(V * C);

    // separate fp32 buffer for embedding grads (scatter-add overflows fp16)
    CUDA_CHECK(cudaMalloc(&dwte_fp32, (size_t)V * C * sizeof(float)));
    drmsf_w = grads.reserve(C);

    drms1_w = new __nv_bfloat16*[L]; dwq = new __nv_bfloat16*[L]; dwk = new __nv_bfloat16*[L];
    dwv = new __nv_bfloat16*[L]; dwo = new __nv_bfloat16*[L]; drms2_w = new __nv_bfloat16*[L];
    dw_gate = new __nv_bfloat16*[L]; dw_up = new __nv_bfloat16*[L]; dw_down = new __nv_bfloat16*[L];

    for (int l = 0; l < L; l++) {
        drms1_w[l] = grads.reserve(C);
        dwq[l]     = grads.reserve(C * C);
        dwk[l]     = grads.reserve(C * kv);
        dwv[l]     = grads.reserve(C * kv);
        dwo[l]     = grads.reserve(C * C);
        drms2_w[l] = grads.reserve(C);
        dw_gate[l] = grads.reserve(C * ff);
        dw_up[l]   = grads.reserve(C * ff);
        dw_down[l] = grads.reserve(ff * C);
    }

    // fp16 activations + fp32 rrms/losses
    size_t n_half_acts = (size_t)BT*C + (size_t)BT*C + (size_t)BT*C + (size_t)BT*kv
                       + (size_t)BT*kv + (size_t)B*nh*T + (size_t)B*nh*T*T + (size_t)BT*C + (size_t)BT*C
                       + (size_t)BT*ff + (size_t)BT*ff + (size_t)BT*ff + (size_t)BT*C
                       + (size_t)BT*C + (size_t)BT*V + (size_t)BT*V
                       + (size_t)BT*C + (size_t)BT*C + (size_t)BT*C + (size_t)BT*C
                       + (size_t)BT*C + (size_t)BT*C + (size_t)BT*V + (size_t)BT*C;
    acts.allocate(n_half_acts);

    x          = acts.reserve(BT * C);
    ln1_out    = acts.reserve(BT * C);
    q          = acts.reserve(BT * C);
    k          = acts.reserve(BT * kv);
    v          = acts.reserve(BT * kv);
    lse        = acts.reserve(B * nh * T);
    att        = acts.reserve(B * nh * T * T);
    attn_out   = acts.reserve(BT * C);
    ln2_out    = acts.reserve(BT * C);
    gate_buf   = acts.reserve(BT * ff);
    up_buf     = acts.reserve(BT * ff);
    hidden_buf = acts.reserve(BT * ff);
    mlp_out    = acts.reserve(BT * C);
    ln_final   = acts.reserve(BT * C);
    logits     = acts.reserve(BT * V);
    probs      = acts.reserve(BT * V);
    dx         = acts.reserve(BT * C);
    dln1_out   = acts.reserve(BT * C);
    dln2_out   = acts.reserve(BT * C);
    dattn_out  = acts.reserve(BT * C);
    dmlp_out   = acts.reserve(BT * C);
    dln_final  = acts.reserve(BT * C);
    dlogits    = acts.reserve(BT * V);
    dx_tmp     = acts.reserve(BT * C);

    // gradient-checkpointing buffers (bf16)
    CUDA_CHECK(cudaMalloc(&x_ckpt, (size_t)L * BT * C * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&x_mid, (size_t)BT * C * sizeof(__nv_bfloat16)));

    // fp32 buffers for rrms and losses (small, need precision)
    CUDA_CHECK(cudaMalloc(&rrms1, BT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rrms2, BT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rrms_final, BT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&losses, BT * sizeof(float)));

    // initialize weights: generate fp32 on CPU → upload to master_params → convert to fp16
    size_t total = n_params;
    float* cpu_buf = (float*)malloc(total * sizeof(float));
    srand(42);
    for (size_t i = 0; i < total; i += 2) {
        float u1 = ((float)rand() / RAND_MAX);
        float u2 = ((float)rand() / RAND_MAX);
        if (u1 < 1e-7f) u1 = 1e-7f;
        float mag = 0.02f * sqrtf(-2.0f * logf(u1));
        cpu_buf[i]     = mag * cosf(2.0f * 3.14159265f * u2);
        if (i + 1 < total) cpu_buf[i + 1] = mag * sinf(2.0f * 3.14159265f * u2);
    }

    // upload fp32 master weights
    CUDA_CHECK(cudaMemcpy(master_params.data, cpu_buf, total * sizeof(float), cudaMemcpyHostToDevice));
    // convert to bf16 and upload
    upload_fp32_as_bf16(params.storage.data, cpu_buf, total);

    // override RMSNorm weights with 1.0
    float* ones = (float*)malloc(C * sizeof(float));
    for (int i = 0; i < C; i++) ones[i] = 1.0f;

    // need to set both master (fp32) and params (fp16) for rmsnorm weights
    // find the offsets manually — rmsf_w is the 3rd reservation
    size_t rmsf_offset = (size_t)V*C + (size_t)V*C;
    CUDA_CHECK(cudaMemcpy(master_params.data + rmsf_offset, ones, C * sizeof(float), cudaMemcpyHostToDevice));
    upload_fp32_as_bf16(rmsf_w, ones, C);

    for (int l = 0; l < L; l++) {
        size_t layer_base = rmsf_offset + C + (size_t)l * (C + (size_t)C*C + (size_t)C*kv + (size_t)C*kv + (size_t)C*C + C + (size_t)C*ff + (size_t)C*ff + (size_t)ff*C);
        // rms1_w is first in each layer block
        CUDA_CHECK(cudaMemcpy(master_params.data + layer_base, ones, C * sizeof(float), cudaMemcpyHostToDevice));
        upload_fp32_as_bf16(rms1_w[l], ones, C);
        // rms2_w is after wq+wk+wv+wo
        size_t rms2_off = layer_base + C + (size_t)C*C + (size_t)C*kv + (size_t)C*kv + (size_t)C*C;
        CUDA_CHECK(cudaMemcpy(master_params.data + rms2_off, ones, C * sizeof(float), cudaMemcpyHostToDevice));
        upload_fp32_as_bf16(rms2_w[l], ones, C);
    }

    ::free(ones);
    ::free(cpu_buf);

    printf("Model (bf16): %d layers, %d heads (%d kv), %d embd, %d ffn\n", L, nh, config.n_kv_head, C, ff);
    printf("Parameters: %.1fM\n", (float)n_params / 1e6f);
    printf("Activation memory: %.1fMB (bf16)\n", (float)(n_half_acts * sizeof(__nv_bfloat16)) / (1024.0f * 1024.0f));
}

void ModelBF16::load(const char* checkpoint_path) {
    size_t total = master_params.numel;
    FILE* f = fopen(checkpoint_path, "rb");
    if (!f) { printf("Error: could not open checkpoint %s\n", checkpoint_path); exit(1); }

    float* cpu_buf = (float*)malloc(total * sizeof(float));
    size_t rd = fread(cpu_buf, sizeof(float), total, f);
    if (rd != total) { printf("Error: checkpoint has %zu floats, expected %zu\n", rd, total); exit(1); }
    fclose(f);

    // load into fp32 master
    CUDA_CHECK(cudaMemcpy(master_params.data, cpu_buf, total * sizeof(float), cudaMemcpyHostToDevice));
    // convert to bf16
    upload_fp32_as_bf16(params.storage.data, cpu_buf, total);
    ::free(cpu_buf);
    printf("Loaded checkpoint (bf16): %s (%zu params)\n", checkpoint_path, total);
}

void ModelBF16::forward(const int* input_ids, const int* targets, int B, int T) {
    int C   = config.n_embd;
    int L   = config.n_layer;
    int nh  = config.n_head;
    int nkv = config.n_kv_head;
    int ff  = config.ffn_hidden;
    int V   = config.vocab_size;
    int BT  = B * T;

    embedding_forward(x, wte, input_ids, B, T, C);

    for (int l = 0; l < L; l++) {
        CUDA_CHECK(cudaMemcpy(x_ckpt + (size_t)l * BT * C, x,
                              (size_t)BT * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        rmsnorm_forward(ln1_out, rrms1, x, rms1_w[l], config.norm_eps, BT, C);
        #ifdef USE_FLASH_ATTN
            attention_forward(attn_out, q, k, v, lse, ln1_out, wq[l], wv[l], wo[l],
                                    B, T, C, nh, nkv, config.rope_theta, cublas_handle)
        #else
            attention_forward(attn_out, q, k, v, att, ln1_out, wq[l], wk[l], wv[l], wo[l],
                            B, T, C, nh, nkv, config.rope_theta, cublas_handle);
        #endif
        residual_forward(x, x, attn_out, BT * C);
        rmsnorm_forward(ln2_out, rrms2, x, rms2_w[l], config.norm_eps, BT, C);
        swiglu_forward(mlp_out, gate_buf, up_buf, hidden_buf,
                       ln2_out, w_gate[l], w_up[l], w_down[l], BT, C, ff, cublas_handle);
        residual_forward(x, x, mlp_out, BT * C);
    }

    rmsnorm_forward(ln_final, rrms_final, x, rmsf_w, config.norm_eps, BT, C);
    matmul_forward(cublas_handle, logits, ln_final, w_out, BT, V, C);
    crossentropy_forward(losses, logits, targets, B, T, V);
    probs = logits;  // crossentropy writes probs in-place
    saved_targets = targets;

    float* cpu_losses = (float*)malloc(BT * sizeof(float));
    CUDA_CHECK(cudaMemcpy(cpu_losses, losses, BT * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < BT; i++) sum += cpu_losses[i];
    loss = sum / (float)BT;
    ::free(cpu_losses);
}

void ModelBF16::backward(int B, int T) {
    int C   = config.n_embd;
    int L   = config.n_layer;
    int nh  = config.n_head;
    int nkv = config.n_kv_head;
    int ff  = config.ffn_hidden;
    int V   = config.vocab_size;
    int BT  = B * T;

    crossentropy_softmax_backward(dlogits, probs, saved_targets, B, T, V, loss_scale);
    matmul_backward(cublas_handle, dln_final, dw_out, dlogits, ln_final, w_out, BT, V, C);
    rmsnorm_backward(dx, drmsf_w, dln_final, x, rmsf_w, rrms_final, BT, C);

    for (int l = L - 1; l >= 0; l--) {
        __nv_bfloat16* x_in = x_ckpt + (size_t)l * BT * C;   // this layer's residual-stream input

        // Recompute layer l's forward activations from the checkpoint.
        rmsnorm_forward(ln1_out, rrms1, x_in, rms1_w[l], config.norm_eps, BT, C);
        attention_forward(attn_out, q, k, v, att,
            ln1_out, wq[l], wk[l], wv[l], wo[l],
            B, T, C, nh, nkv, config.rope_theta, cublas_handle);
        residual_forward(x_mid, x_in, attn_out, BT * C);
        rmsnorm_forward(ln2_out, rrms2, x_mid, rms2_w[l], config.norm_eps, BT, C);
        swiglu_forward(mlp_out, gate_buf, up_buf, hidden_buf,
                       ln2_out, w_gate[l], w_up[l], w_down[l], BT, C, ff, cublas_handle);

        CUDA_CHECK(cudaMemcpy(dx_tmp, dx, BT * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(dmlp_out, 0, BT * C * sizeof(__nv_bfloat16)));
        residual_backward(dx, dmlp_out, dx_tmp, BT * C);

        swiglu_backward(dln2_out, dw_gate[l], dw_up[l], dw_down[l],
                        dmlp_out, ln2_out, gate_buf, up_buf,
                        w_gate[l], w_up[l], w_down[l], BT, C, ff, cublas_handle);

        rmsnorm_backward(dx_tmp, drms2_w[l], dln2_out, x_mid, rms2_w[l], rrms2, BT, C);  // input was x_mid
        residual_forward(dx, dx, dx_tmp, BT * C);

        CUDA_CHECK(cudaMemcpy(dx_tmp, dx, BT * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(dattn_out, 0, BT * C * sizeof(__nv_bfloat16)));
        residual_backward(dx, dattn_out, dx_tmp, BT * C);

        #ifdef USE_FLASH_ATTN
            attention_backward(dln1_out, dwq[l], dwk[l], dwv[l], dwo[l],
                dattn_out, ln1_out, q, k, v, lse,
                wq[l], wk[l], wv[l], wo[l],
                B, T, C, nh, nkv, config.rope_theta, cublas_handle);
        #else
            attention_backward(dln1_out, dwq[l], dwk[l], dwv[l], dwo[l],
                dattn_out, ln1_out, q, k, v, att,
                wq[l], wk[l], wv[l], wo[l],
                B, T, C, nh, nkv, config.rope_theta, cublas_handle);
        #endif

        rmsnorm_backward(dx_tmp, drms1_w[l], dln1_out, x_in, rms1_w[l], rrms1, BT, C);  // input was x_in
        residual_forward(dx, dx, dx_tmp, BT * C);
    }
}

void ModelBF16::zero_grad() {
    CUDA_CHECK(cudaMemset(grads.storage.data, 0, grads.storage.bytes()));
    CUDA_CHECK(cudaMemset(dwte_fp32, 0, (size_t)config.vocab_size * config.n_embd * sizeof(float)));
}

void ModelBF16::update(float lr, TrainConfig tcfg, int step, float grad_clip_scale) {
    int N = (int)params.storage.numel;
    int emb_N = config.vocab_size * config.n_embd;  // V*C elements for embedding
    float grad_scale = (1.0f / loss_scale) * grad_clip_scale;

    // Scale fp32 embedding grads on GPU, run fp32 AdamW, then sync fp16 copy
    if (grad_scale != 1.0f) scale_grads(dwte_fp32, grad_scale, emb_N);
    adamw_update(master_params.data, dwte_fp32, adam_m.data, adam_v.data,
                 lr, tcfg.beta1, tcfg.beta2, tcfg.eps, tcfg.weight_decay, step, emb_N);
    fp32_to_bf16(master_params.data, params.storage.data, emb_N);

    // 2) update all other params using fp16 grads (skip first emb_N elements)
    int rest_N = N - emb_N;
    adamw_update(params.storage.data + emb_N, master_params.data + emb_N,
                 grads.storage.data + emb_N,
                 adam_m.data + emb_N, adam_v.data + emb_N,
                 lr, tcfg.beta1, tcfg.beta2, tcfg.eps,
                 tcfg.weight_decay, grad_scale, step, rest_N);
}

void ModelBF16::free() {
    cublasDestroy(cublas_handle);
    params.free();
    grads.free();
    acts.free();
    master_params.free();
    adam_m.free();
    adam_v.free();

    CUDA_CHECK(cudaFree(rrms1));
    CUDA_CHECK(cudaFree(rrms2));
    CUDA_CHECK(cudaFree(rrms_final));
    CUDA_CHECK(cudaFree(losses));
    CUDA_CHECK(cudaFree(dwte_fp32));
    CUDA_CHECK(cudaFree(x_ckpt));
    CUDA_CHECK(cudaFree(x_mid));

    delete[] rms1_w;  delete[] wq;     delete[] wk;
    delete[] wv;      delete[] wo;     delete[] rms2_w;
    delete[] w_gate;  delete[] w_up;   delete[] w_down;
    delete[] drms1_w; delete[] dwq;    delete[] dwk;
    delete[] dwv;     delete[] dwo;    delete[] drms2_w;
    delete[] dw_gate; delete[] dw_up;  delete[] dw_down;
}
