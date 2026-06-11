# Full Backward Pass Order

The complete backward pass through the entire model, showing
every operation in the exact order it executes.

Config: B=4, T=512, C=768, n_head=12, n_kv_head=4, head_dim=64, ffn=2048, V=32000.

---

## Execution Order

```
  The backward pass traverses the model in REVERSE order.

  ═══════════════════════════════════════════════════════════════════
  STEP    OPERATION               INPUT GRAD    WEIGHT GRADS
  ═══════════════════════════════════════════════════════════════════

  1. Cross-Entropy + Softmax
     dlogits = probs - one_hot      —              —

  2. Output projection backward
     dhidden = dlogits @ W_out^T    dW_out         [N,C]

  3. Final RMSNorm backward         dγ_final       [N,C]

  ─── For layer L = 11 down to 0: ───────────────────────────────

  4. Residual split (post-MLP)
     dx_mlp = dout                  —              —
     dx_residual = dout             —              —

  5. RMSNorm backward (pre-MLP)    dγ_mlp_L       [N,C]

  6. SwiGLU backward
     6a. Down projection            dW_down_L      [N,ffn]
     6b. Gate × up backward         —              [N,ffn]
     6c. SiLU backward              —              [N,ffn]
     6d. Gate projection             dW_gate_L      [N,C]
     6e. Up projection               dW_up_L        [N,C]

  7. Residual add (dmlp + dresidual) —             —

  8. Residual split (post-attention)
     dx_attn = dout                 —              —
     dx_residual = dout             —              —

  9. RMSNorm backward (pre-attn)   dγ_attn_L      [N,C]

  10. Attention backward
      10a. Output proj backward      dW_o_L         [N,C]
      10b. Per-head: probs@V back    —              [T,T],[T,64]
      10c. Softmax backward          —              [T,T]
      10d. Score backward (Q@K^T)    —              [T,64]
      10e. RoPE backward             —              —
      10f. Q,K,V proj backward       dW_q,dW_k,dW_v [N,C]

  11. Residual add (dattn + dresidual) —            —

  ─── End layer loop ────────────────────────────────────────────

  12. Embedding backward
      dW_e via scatter-add          dW_e           —

  ═══════════════════════════════════════════════════════════════════
  TOTAL weight gradients per step:
    Embedding:  1 table           (V × C = 24.6M grads)
    Per layer:  W_q, W_k, W_v, W_o, W_gate, W_up, W_down, 2×γ
                (9 weight tensors × 12 layers)
    Output:     W_out + γ_final   (24.6M + 768 grads)
  ═══════════════════════════════════════════════════════════════════
```

---

## FLOPs Breakdown

```
  Total backward FLOPs (dominated by matmuls):

  ┌─────────────────────────────────────────────────────────────┐
  │  Component            Forward GFLOPs   Backward GFLOPs     │
  │  ──────────────────   ──────────────   ──────────────      │
  │  Embedding            0                0.002               │
  │  Attention × 12       97               194                 │
  │  SwiGLU × 12          232              464                 │
  │  RMSNorm × 25         0.16             0.47                │
  │  Residual × 24        0.04             0                   │
  │  RoPE × 12            0.003            0.003               │
  │  Output proj          101              201                 │
  │  Softmax (all)        1.1              1.1                 │
  │  ─────────────────────────────────────────────────────      │
  │  TOTAL                ~431 GFLOPs      ~861 GFLOPs         │
  │  GRAND TOTAL:         ~1,292 GFLOPs per training step      │
  │                                                             │
  │  At 101 TFLOPS BF16:  ~12.8 ms (theoretical minimum)      │
  │  At ~30% utilization:  ~43 ms per step                     │
  │  At 4.9k tok/s with 2048 tok/step: ~418 ms per step       │
  └─────────────────────────────────────────────────────────────┘
```

---

## Visual: Forward vs Backward Data Flow

```
  FORWARD (top to bottom):
  ═══════════════════════════════════════════════════

  tokens [2048]
      │
      ▼
  Embedding lookup → hidden [2048, 768]
      │
      ▼
  ┌─ Layer 0 ──────────────────────────────────────┐
  │  RMSNorm → Attention → Residual Add            │
  │  RMSNorm → SwiGLU    → Residual Add            │
  └────────────────────────────────────────────────┘
      │
      ▼
  ... (layers 1-10) ...
      │
      ▼
  ┌─ Layer 11 ─────────────────────────────────────┐
  │  RMSNorm → Attention → Residual Add            │
  │  RMSNorm → SwiGLU    → Residual Add            │
  └────────────────────────────────────────────────┘
      │
      ▼
  Final RMSNorm → Output Projection → Softmax → Loss


  BACKWARD (bottom to top):
  ═══════════════════════════════════════════════════

  Loss = 7.63
      │
      ▼
  dlogits = probs - one_hot    ← backprop STARTS here
      │
      ▼
  Output Projection backward → Final RMSNorm backward
      │
      ▼
  ┌─ Layer 11 backward ────────────────────────────┐
  │  Residual split → SwiGLU back → RMSNorm back   │
  │  Residual split → Attention back → RMSNorm back │
  └────────────────────────────────────────────────┘
      │
      ▼
  ... (layers 10-1) ...
      │
      ▼
  ┌─ Layer 0 backward ────────────────────────────┐
  │  Residual split → SwiGLU back → RMSNorm back   │
  │  Residual split → Attention back → RMSNorm back │
  └────────────────────────────────────────────────┘
      │
      ▼
  Embedding backward (scatter-add)    ← backprop ENDS here
      │
      ▼
  Gradient clipping → AdamW update    ← optimizer runs
```

---

## File Cross-References

```
  Each component's detailed backward derivation:

  chain_rule.md          ← foundation for everything
  crossentropy_backprop.md  ← where gradients originate
  matmul_backprop.md     ← used by every projection
  residual_backprop.md   ← gradient highway
  rmsnorm_backprop.md    ← trickiest element-wise backward
  attention_backprop.md  ← most complex (GQA + softmax)
  rope_backprop.md       ← simplest (just flip sin sign)
  swiglu_backprop.md     ← most compute-heavy (6 matmuls)
  softmax_backprop.md    ← O(n) trick avoids Jacobian
  embedding_backprop.md  ← end of chain (scatter-add)
  adamw_backprop.md      ← post-backprop optimizer
```
