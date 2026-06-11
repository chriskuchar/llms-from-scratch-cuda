#pragma once
#include "tensor.h"
#include <cstdio>
#include <cstdlib>

// Reads pre-tokenized data from a binary file of int32 token IDs.
// next_batch() returns (input_ids, targets) on GPU, where targets =
// input_ids shifted by one position (next-token prediction).
// File format: raw int32s, e.g. python tokens.astype(np.int32).tofile("train.bin")
struct DataLoader {
    int B;
    int T;
    int* data;          // all tokens, in CPU memory
    size_t n_tokens;
    size_t pos;         // current read position in the token stream

    int* input_ids_gpu; // [B, T] on GPU
    int* targets_gpu;   // [B, T] on GPU

    void open(const char* filename,
              int batch_size,
              int seq_len);

    // returns false if not enough tokens left (epoch is done)
    bool next_batch();

    void reset();

    void free();
};
