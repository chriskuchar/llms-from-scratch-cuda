#include "dataloader.h"
#include <cstdio>
#include <cstdlib>

// Load a binary file of int32 token IDs into CPU memory.
void DataLoader::open(const char* filename,
                      int batch_size,
                      int seq_len) {
    B = batch_size;
    T = seq_len;
    pos = 0;

    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Error: could not open %s\n", filename); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    n_tokens = file_size / sizeof(int);

    data = (int*)malloc(file_size);
    size_t read = fread(data, sizeof(int), n_tokens, f);
    if (read != n_tokens) { printf("Error: read %zu tokens, expected %zu\n", read, n_tokens); exit(1); }
    fclose(f);

    printf("DataLoader: %zu tokens from %s\n", n_tokens, filename);
    printf("  %zu batches per epoch (B=%d, T=%d)\n", n_tokens / (B * T), B, T);

    CUDA_CHECK(cudaMalloc(&input_ids_gpu, B * T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&targets_gpu,   B * T * sizeof(int)));
}

// Copy the next B*T tokens to GPU and advance position; targets are shifted by 1.
bool DataLoader::next_batch() {
    // Need B*T + 1 tokens: B*T for input, plus 1 for the shifted targets.
    if (pos + B * T + 1 > n_tokens) {
        return false;
    }

    CUDA_CHECK(cudaMemcpy(input_ids_gpu, data + pos,
                          B * T * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(targets_gpu, data + pos + 1,
                          B * T * sizeof(int), cudaMemcpyHostToDevice));

    pos += B * T;

    return true;
}

void DataLoader::reset() {
    pos = 0;
}

void DataLoader::free() {
    ::free(data);
    CUDA_CHECK(cudaFree(input_ids_gpu));
    CUDA_CHECK(cudaFree(targets_gpu));
}
