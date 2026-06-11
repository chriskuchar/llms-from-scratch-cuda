#include "model.h"
#include "dataloader.h"
#include "kernels.cuh"
#include <cstdio>
#include <cmath>
#include <ctime>

// Learning rate schedule: linear warmup → cosine decay.
float get_lr(int step,
             float peak_lr,
             float min_lr,
             int warmup_steps,
             int max_steps) {
    if (step < warmup_steps) {
        return peak_lr * ((float)step / (float)warmup_steps);
    }
    float progress = (float)(step - warmup_steps) / (float)(max_steps - warmup_steps);
    if (progress > 1.0f) progress = 1.0f;
    float decay = 0.5f * (1.0f + cosf(3.14159265f * progress));
    return min_lr + (peak_lr - min_lr) * decay;
}

// Gradient clipping: scale all gradients if global L2 norm > max_norm.
void clip_gradients(float* grads,
                    int N,
                    float max_norm) {
    // Norm is computed on CPU for correctness-first simplicity (could be a GPU reduce).
    float* cpu_grads = (float*)malloc(N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(cpu_grads, grads, N * sizeof(float), cudaMemcpyDeviceToHost));

    double sum_sq = 0.0;
    for (int i = 0; i < N; i++) {
        sum_sq += (double)cpu_grads[i] * (double)cpu_grads[i];
    }
    float global_norm = (float)sqrt(sum_sq);

    if (global_norm > max_norm) {
        float scale = max_norm / global_norm;
        for (int i = 0; i < N; i++) {
            cpu_grads[i] *= scale;
        }
        CUDA_CHECK(cudaMemcpy(grads, cpu_grads, N * sizeof(float), cudaMemcpyHostToDevice));
    }

    free(cpu_grads);
}

int main(int argc, char** argv) {
    ModelConfig model_cfg;
    TrainConfig train_cfg;

    // Usage: ./train [data_file] [--checkpoint path] [--lr 3e-4] [--steps 30000] [--bf16]
    const char* train_file = "data/train.bin";
    const char* checkpoint_path = nullptr;
    bool use_bf16 = false;
    int start_step = 0;                        // offset for LR schedule when resuming from checkpoint
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            train_cfg.learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            train_cfg.max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--accum") == 0 && i + 1 < argc) {
            train_cfg.accum_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            train_cfg.warmup_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bf16") == 0) {
            use_bf16 = true;
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            train_cfg.batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            train_cfg.save_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--start-step") == 0 && i + 1 < argc) {
            start_step = atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            train_file = argv[i];              // positional arg = data file
        }
    }

    int B = train_cfg.batch_size;
    int T = train_cfg.seq_len;
    int accum = train_cfg.accum_steps;
    int tokens_per_step = B * T * accum;

    printf("Building model%s...\n", use_bf16 ? " (bf16 mixed precision)" : " (fp32)");

    // Both model types share the same interface, so we dispatch on use_bf16.
    Model model_fp32;
    ModelBF16 model_bf16;

    if (use_bf16) {
        model_bf16.build(model_cfg, B, T);
        if (checkpoint_path) model_bf16.load(checkpoint_path);
        if (checkpoint_path) {
            size_t n = model_bf16.master_params.numel;
            FILE* f = fopen(checkpoint_path, "rb");
            fseek(f, n * sizeof(float), SEEK_SET);
            int magic = 0;
            if (fread(&magic, sizeof(int), 1, f) == 1 && magic == 0x4F50544D){
                int saved_step = 0;
                fread(&saved_step, sizeof(int), 1, f);
                float* cpu_buf = (float*)malloc(n*sizeof(float));
                fread(cpu_buf, sizeof(float), n, f);
                CUDA_CHECK(cudaMemcpy(model_bf16.adam_m.data, cpu_buf, n * sizeof(float), cudaMemcpyHostToDevice));
                fread(cpu_buf, sizeof(float), n, f);
                CUDA_CHECK(cudaMemcpy(model_bf16.adam_v.data, cpu_buf, n * sizeof(float), cudaMemcpyHostToDevice));
                ::free(cpu_buf);
                if (start_step == 0) start_step = saved_step;
                printf("Restored optimizer state from step %d\n", saved_step);
            }
            fclose(f);
        }
    } else {
        model_fp32.build(model_cfg, B, T);
        if (checkpoint_path) model_fp32.load(checkpoint_path);
        if (checkpoint_path) {
            size_t n = model_fp32.params.storage.numel;
            FILE* f = fopen(checkpoint_path, "rb");
            fseek(f, n * sizeof(float), SEEK_SET);
            int magic = 0;
            if (fread(&magic, sizeof(int), 1, f) == 1 && magic == 0x4F50544D){
                int saved_step = 0;
                fread(&saved_step, sizeof(int), 1, f);
                float* cpu_buf = (float*)malloc(n*sizeof(float));
                fread(cpu_buf, sizeof(float), n, f);
                CUDA_CHECK(cudaMemcpy(model_fp32.adam_m.data, cpu_buf, n * sizeof(float), cudaMemcpyHostToDevice));
                fread(cpu_buf, sizeof(float), n, f);
                CUDA_CHECK(cudaMemcpy(model_fp32.adam_v.data, cpu_buf, n * sizeof(float), cudaMemcpyHostToDevice));
                ::free(cpu_buf);
                if (start_step == 0) start_step = saved_step;
                printf("Restored optimizer state from step %d\n", saved_step);
            }
            fclose(f);
        }
    }

    printf("Opening data: %s\n", train_file);
    DataLoader loader;
    loader.open(train_file, B, T);

    printf("Training for %d steps (accum=%d, effective batch=%d tokens)...\n",
           train_cfg.max_steps, accum, tokens_per_step);
    if (start_step > 0) {
        printf("Resuming LR schedule from step %d\n", start_step);
    }
    clock_t start = clock();

    for (int step = 1; step <= train_cfg.max_steps; step++) {
        int global_step = step + start_step;

        float lr = get_lr(global_step, train_cfg.learning_rate, train_cfg.min_lr,
                          train_cfg.warmup_steps, train_cfg.max_steps + start_step);

        if (use_bf16) model_bf16.zero_grad();
        else model_fp32.zero_grad();

        float accum_loss = 0.0f;
        for (int micro = 0; micro < accum; micro++) {
            if (!loader.next_batch()) {
                loader.reset();
                loader.next_batch();
            }

            if (use_bf16) {
                model_bf16.forward(loader.input_ids_gpu, loader.targets_gpu, B, T);
                model_bf16.backward(B, T);
                // use fp32 embedding backward to avoid scatter-add overflow in fp16
                embedding_backward(model_bf16.dwte_fp32, model_bf16.dx, loader.input_ids_gpu,
                                   B, T, model_bf16.config.n_embd);
                accum_loss += model_bf16.loss;
            } else {
                model_fp32.forward(loader.input_ids_gpu, loader.targets_gpu, B, T);
                model_fp32.backward(B, T);
                embedding_backward(model_fp32.dwte, model_fp32.dx, loader.input_ids_gpu,
                                   B, T, model_fp32.config.n_embd);
                accum_loss += model_fp32.loss;
            }
        }
        accum_loss /= accum;

        if (use_bf16) {
            int N = (int)model_bf16.params.storage.numel;
            int emb_N = model_bf16.config.vocab_size * model_bf16.config.n_embd;

            // GPU-side norm: fp32 for embedding, fp16 for rest (sanitizes inf/nan in-place)
            float global_norm = compute_grad_norm_bf16(
                model_bf16.dwte_fp32, emb_N,
                model_bf16.grads.storage.data + emb_N, N - emb_N,
                model_bf16.loss_scale);

            float clip_scale = (global_norm > train_cfg.grad_clip)
                             ? train_cfg.grad_clip / global_norm : 1.0f;

            model_bf16.update(lr, train_cfg, global_step, clip_scale);
        } else {
            clip_gradients(model_fp32.grads.storage.data, (int)model_fp32.params.storage.numel,
                           train_cfg.grad_clip);
            model_fp32.update(lr, train_cfg, global_step);
        }

        if (step % train_cfg.log_interval == 0 || step == 1) {
            clock_t now = clock();
            float elapsed = (float)(now - start) / CLOCKS_PER_SEC;
            float tokens_so_far = (float)step * tokens_per_step;
            float tokens_per_sec = tokens_so_far / elapsed;
            printf("step %5d | loss %.4f | lr %.2e | %.0f tok/s\n",
                   global_step, accum_loss, lr, tokens_per_sec);
        }

        // Save checkpoint (always fp32 master weights for compatibility).
        if (step % train_cfg.save_interval == 0) {
            char path[256];
            snprintf(path, sizeof(path), "checkpoints/step_%d.bin", global_step);
            printf("Saving checkpoint: %s\n", path);

            float* cpu_params;
            size_t n;
            if (use_bf16) {
                n = model_bf16.master_params.numel;
                cpu_params = (float*)malloc(n * sizeof(float));
                CUDA_CHECK(cudaMemcpy(cpu_params, model_bf16.master_params.data,
                                      n * sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                n = model_fp32.params.storage.numel;
                cpu_params = (float*)malloc(n * sizeof(float));
                CUDA_CHECK(cudaMemcpy(cpu_params, model_fp32.params.storage.data,
                                      n * sizeof(float), cudaMemcpyDeviceToHost));
            }
            FILE* f = fopen(path, "wb");
            if (f) {
                fwrite(cpu_params, sizeof(float), n, f);
                int magic = 0x4F50544d;
                fwrite(&magic, sizeof(int), 1, f);
                fwrite(&global_step, sizeof(int), 1, f);
                float* cpu_opt = (float*)malloc(n * sizeof(float));
                if(use_bf16){
                    CUDA_CHECK(cudaMemcpy(cpu_opt, model_bf16.adam_m.data, n * sizeof(float), cudaMemcpyDeviceToHost));
                    fwrite(cpu_opt, sizeof(float), n, f);
                    CUDA_CHECK(cudaMemcpy(cpu_opt, model_bf16.adam_v.data, n * sizeof(float), cudaMemcpyDeviceToHost));
                    fwrite(cpu_opt, sizeof(float), n, f);
                } else { 
                    CUDA_CHECK(cudaMemcpy(cpu_opt, model_fp32.adam_m.data, n * sizeof(float), cudaMemcpyDeviceToHost));
                    fwrite(cpu_opt, sizeof(float), n, f);
                    CUDA_CHECK(cudaMemcpy(cpu_opt, model_fp32.adam_v.data, n * sizeof(float), cudaMemcpyDeviceToHost));
                    fwrite(cpu_opt, sizeof(float), n, f);                    
                }
                ::free(cpu_opt);
                fclose(f);
            } else {
                printf("Warning: could not write checkpoint %s\n", path);
            }
            ::free(cpu_params);
        }
    }

    printf("Training complete.\n");
    loader.free();
    if (use_bf16) model_bf16.free();
    else model_fp32.free();
    return 0;
}
