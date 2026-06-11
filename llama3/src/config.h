
#pragma once

// Architecture config for a LLaMA 3-style transformer. Same block as
// ../llama2-mistral (GQA + RoPE + RMSNorm + SwiGLU + pre-norm); the only deltas
// are the 3 values marked "LLaMA 3": 128K tokenizer, higher RoPE base, longer context.
struct ModelConfig {
    int vocab_size = 128256;      // LLaMA 3: 128K tiktoken-style BPE (LLaMA 2: 32K SentencePiece)
    int n_layer = 12;
    int n_head = 12;
    int n_kv_head = 4;            // GQA: every 3 query heads share 1 KV head
    int n_embd = 768;             // embedding dimension (C)
    int max_seq_len = 8192;       // LLaMA 3: 8K base context (3.1 extends to 128K)
    int ffn_hidden = 2048;        // SwiGLU intermediate size (≈ 8/3 * n_embd)
    float rope_theta = 500000.0f; // LLaMA 3: RoPE base 500,000 (LLaMA 2: 10,000)
    float norm_eps = 1e-5f;

    int head_dim() const { return n_embd / n_head; }

    int kv_dim() const { return n_kv_head * head_dim(); }

    // query heads per KV head, used to broadcast K,V across head groups
    int gqa_ratio() const { return n_head / n_kv_head; }

    // Total learnable parameters; sizes the flat weight buffer.
    long long n_params() const {
        int C = n_embd;
        int kv = kv_dim();
        int ff = ffn_hidden;

        long long emb = (long long)vocab_size * C;   // wte:   [V × C]
        long long out = (long long)vocab_size * C;    // w_out: [V × C] (no weight tying)

        long long attn = (long long)n_layer * (       // per layer:
            (long long)C * C                          //   W_q  [C × C]
            + (long long)C * kv                       //   W_k  [C × kv]
            + (long long)C * kv                       //   W_v  [C × kv]
            + (long long)C * C                        //   W_o  [C × C]
            + 2LL * C                                 //   2× RMSNorm [C]
        );

        long long mlp = (long long)n_layer * (        // per layer:
            (long long)C * ff                         //   W_gate [C × ff]
            + (long long)C * ff                       //   W_up   [C × ff]
            + (long long)ff * C                       //   W_down [ff × C]
        );

        long long norm = C;                           // final RMSNorm
        // ≈ 272M params: the 128K vocab makes embedding + output (~197M) dominate
        // at this small n_embd — the cost of the LLaMA 3 tokenizer at 125M-class scale.
        return emb + out + attn + mlp + norm;
    }
};

// Training hyperparameters.
struct TrainConfig {
    float learning_rate = 3e-4f;  // peak LR, reached at end of warmup
    float min_lr        = 3e-5f;  // LR floor at end of cosine decay
    float beta1         = 0.9f;
    float beta2         = 0.95f;  // lower than default 0.999 for LLM stability
    float eps           = 1e-8f;
    float weight_decay  = 0.1f;   // decoupled (AdamW, not L2)
    float grad_clip     = 1.0f;   // max global gradient norm

    int batch_size      = 4;      // B
    int seq_len         = 512;    // T, must be <= max_seq_len
    int accum_steps     = 16;     // effective batch = B * T * accum_steps
    int max_steps       = 30000;
    int warmup_steps    = 2000;
    int log_interval    = 10;
    int save_interval   = 2000;
    int eval_interval   = 500;
    int seed            = 42;
};
