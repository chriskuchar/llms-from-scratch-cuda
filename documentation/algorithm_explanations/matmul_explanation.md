# Matrix Multiplication — The Core Operation

## Core Formula

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Forward:                                                           │
│    C = A × B                                                       │
│    C[i, j] = Σ_k  A[i,k] × B[k,j]                               │
│                                                                     │
│    A ∈ ℝ^{M×K}    B ∈ ℝ^{K×N}    C ∈ ℝ^{M×N}                    │
│                                                                     │
│         K cols              N cols                N cols            │
│     ┌───────────┐       ┌───────────┐        ┌───────────┐        │
│  M  │     A     │   ×   │     B     │   =  M │     C     │        │
│     │  [M × K]  │    K  │  [K × N]  │        │  [M × N]  │        │
│     └───────────┘       └───────────┘        └───────────┘        │
│                                                                     │
│  Each output element = dot product of one row of A × one col of B  │
│                                                                     │
│  Backward:                                                          │
│    dA = dC × B^T          gradient for input   [M,N]@[N,K]=[M,K] │
│    dB = A^T × dC          gradient for weights  [K,M]@[M,N]=[K,N] │
│                                                                     │
│  cuBLAS call (row-major trick — swap A,B and M,N):                 │
│    cublasSgemm(handle, N, M, K, &α, B, N, A, K, &β, C, N)        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Breaking Down Each Part

```
C[i,j] = Σ_k A[i,k]×B[k,j]   What it does: dot product of row i with column j
                                Row of A = one token's features
                                Column of B = one output neuron's weights
                                Dot product = "how much does this token activate
                                              this output neuron?"

dA = dC × B^T                  What it does: gradient for the input
                                "How should we change A to reduce the loss?"
                                Each row of A contributed to the entire row of C,
                                so we need all columns of B^T to compute dA.

dB = A^T × dC                  What it does: gradient for the weights
                                "How should we change B to reduce the loss?"
                                Each column of B contributed to the entire
                                column of C, so we need all rows of A^T.

Backward = 2 × Forward         Every forward matmul produces TWO backward matmuls
                                This is why backward is ~2× the compute of forward.
```

---

## What It Does

Matrix multiplication (matmul) is the single most common operation in the
model. Nearly every layer is a matmul:

```
Q projection:        input @ W_q           matmul
K projection:        input @ W_k           matmul
V projection:        input @ W_v           matmul
attention scores:    Q @ K^T               matmul
attention output:    probs @ V             matmul
output projection:   context @ W_o         matmul
gate projection:     input @ W_gate        matmul
up projection:       input @ W_up          matmul
down projection:     hidden @ W_down       matmul
final logits:        hidden @ W_out        matmul

About 90% of the model's compute is matrix multiplication.
```

---

## cuBLAS: Let NVIDIA Do It

```
  Why not write our own kernel?

  Naive matmul:    ~1% of peak GPU throughput
  Optimized needs: tiling, register blocking, coalescing,
                   pipelining, Tensor Core utilization
  cuBLAS:          ~80-90% of peak   ← decades of NVIDIA optimization
```

---

## Row-Major vs Column-Major: The cuBLAS Trick

```
  Problem:
    C/C++ stores matrices row-major:   [row0 | row1 | row2 | ...]
    cuBLAS expects column-major:       [col0 | col1 | col2 | ...]
    cuBLAS sees our matrix TRANSPOSED.

  Solution:
    We want:       C = A × B        (row-major)
    cuBLAS sees:   C^T = B^T × A^T  (column-major)

    So we swap A and B, swap M and N:

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,     no explicit transposes
        N, M, K,                       dimensions swapped
        &alpha,
        B, N,                          B first (not A!)
        A, K,                          A second
        &beta,
        C, N);

    Takeaway: swap A↔B, swap M↔N, and it just works.
```

---

## Tensor Cores (BF16 Path)

```
┌────────────────────────────────────────────────────────────────┐
│  cublasGemmEx — activates Tensor Cores                        │
│                                                                │
│  cublasGemmEx(handle,                                          │
│      CUBLAS_OP_N, CUBLAS_OP_N,                                 │
│      N, M, K,                                                  │
│      &alpha,                                                   │
│      B, CUDA_R_16BF, N,       ← BF16 input                   │
│      A, CUDA_R_16BF, K,       ← BF16 input                   │
│      &beta,                                                    │
│      C, CUDA_R_16BF, N,       ← BF16 output                  │
│      CUBLAS_COMPUTE_32F,       ← FP32 accumulation inside     │
│      CUBLAS_GEMM_DEFAULT);     ← let cuBLAS pick algorithm    │
│                                                                │
│  What Tensor Cores do:                                         │
│    Regular CUDA cores:  4×4 matmul = 64 multiply-adds          │
│                         = 64 clock cycles                      │
│    Tensor Cores:        4×4 matmul = 64 multiply-adds          │
│                         = 1 clock cycle    ← 64× faster        │
│                                                                │
│  RTX 3060 (SM 86, Ampere):                                     │
│    CUDA cores:    3584 @ 1.78 GHz = 12.7 TFLOPS (FP32)        │
│    Tensor Cores:  112 @ 1.78 GHz  = 101 TFLOPS (BF16)  ~8×    │
│                                                                │
│  Why CUBLAS_COMPUTE_32F?                                        │
│    A(BF16) × B(BF16) → accumulated in FP32 → stored as BF16   │
│    BF16 has ~3 decimal digits. With K=768 multiply-adds,       │
│    FP32 accumulation prevents rounding errors from snowballing. │
│    No speed penalty — hardware does it natively.                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Compute Budget

```
For a single forward pass through the full model:

Layer                    Shape                          FLOPs
──────────────────────   ─────────────────────────      ──────────
Q projection             [2048,768] @ [768,768]         2.4B
K projection             [2048,768] @ [768,256]         0.8B
V projection             [2048,768] @ [768,256]         0.8B
Attention (Q@K^T)        [2048,12,512] @ [512,64]       0.8B
Attention (probs@V)      [2048,12,512] @ [512,64]       0.8B
Output projection        [2048,768] @ [768,768]         2.4B
Gate projection          [2048,768] @ [768,2048]         6.4B
Up projection            [2048,768] @ [768,2048]         6.4B
Down projection          [2048,2048] @ [2048,768]        6.4B
──────────────────────                                  ──────
Per layer total:                                        ~27B FLOPs
× 12 layers:                                           ~324B FLOPs
Final projection         [2048,768] @ [768,32000]       100B FLOPs
──────────────────────                                  ──────
Forward total:                                         ~424B FLOPs
Backward (2× forward):                                ~848B FLOPs
──────────────────────                                  ──────
Per step total:                                        ~1.3T FLOPs
```

---

## alpha and beta: Accumulation Mode

```
  cuBLAS computes: C = α × (A @ B) + β × C

  α=1.0, β=0.0  →  C = A @ B           overwrite C
  α=1.0, β=1.0  →  C = A @ B + C       accumulate into C
  α=2.0, β=0.0  →  C = 2 × (A @ B)     scaled multiply
  α=1.0, β=-1.0 →  C = A @ B - C       subtract

  Our code always uses α=1.0, β=0.0 (simple overwrite).
```
