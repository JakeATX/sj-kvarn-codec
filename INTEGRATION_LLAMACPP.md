# Sketch: SJ-KVaRN in llama.cpp (ggml registration)

This is a plan for a llama.cpp pull request, not a patch. It lists what has to be
registered in ggml and in `llama-kv-cache` so that the cache stores SJ-KVaRN and attention
reads it. The CPU parts can call `sj_kvarn.h` directly. A CUDA backend can start from
`sj_kvarn_cuda.cuh` and replace kernels one at a time, checking each against the
reference.

The layout below has been run end to end in a llama.cpp tree with a CPU and a CUDA
backend. The names used here (`SJKVARN`) are suggestions for an upstream PR.

## 1. Types

| ggml type | Block | Size | Use |
|---|---|---|---|
| `GGML_TYPE_TQ6_0` (new) | 128 values | 98 bytes: `{ggml_half norm; uint8_t qs[64]; uint8_t qh[32];}` | staging ring rows (tail) |
| `GGML_TYPE_F16` | | | sink rows (positions `[0, 128)`) |
| `GGML_TYPE_I8` | | | body pool: opaque sealed records |

There is no ggml type for the body. A record covers a whole (128-token group, head) tile
and is not a row-wise block, so the body is an I8 pool tensor per layer of size
`rec_bytes * n_head_kv * n_groups`, with records ordered by (group, head). This keeps
the body out of `ggml_type_traits` and out of `mul_mat`.

Entry for `TQ6_0` in `ggml.c` `type_traits`:

```c
[GGML_TYPE_TQ6_0] = {
    .type_name      = "tq6_0",
    .blck_size      = 128,
    .type_size      = 98,
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t)   dequantize_row_tq6_0,      // sj_kvarn_tq6_dequantize_row
    .from_float_ref = (ggml_from_float_t) quantize_row_tq6_0_ref,    // sj_kvarn_tq6_quantize_row
},
```

`TQ6_0` is a KV-cache-only type, so `ggml_quantize_chunk` and the GGUF weight paths can
reject it. Add it to `llama_model_loader`'s list of rejected types and to the
`--cache-type-k/v` parser.

## 2. Ops

| Op | Signature (sketch) | Notes |
|---|---|---|
| Hadamard | `ggml_hadamard(ctx, a, n)` | Orthonormal Sylvester transform along `ne[0]` in blocks of `n = head_dim`. Its own inverse. Used on Q, K and V and on the attention output. The rotation can also be folded into the Q/K/V/O weights at load time; that is not part of this sketch. |
| staged write | `ggml_set_rows_stage(ctx, dst, src, rows)` | Like `ggml_set_rows`, but for a `TQ6_0` destination the input is already rotated (no rotation inside the quantizer). For positions below the sink, K and V are written as fp16 into the sink area. |
| seal | `ggml_sjkvarn_seal_dyn(ctx, body, k_ring, v_ring, desc, D, G, bits_k, bits_v, iters, n_groups_max)` | Reads `desc` at run time and seals the groups covering `[B_old, B)` from the ring into records. Each row is dequantized from `TQ6_0`, rounded to fp16 and then sealed (`sj_kvarn_seal_group`). `n_groups_max` fixes the launch size so the graph can be reused. Body type in `op_params` (scalar or trellis). |
| attention | `ggml_flash_attn_ext_set_sjkvarn(fa, body, desc, bits_k, bits_v, n_kv_pad)` | Marks an existing `flash_attn_ext` node as region-aware. `k`/`v` (`src[1]`, `src[2]`) stay the sink + ring tensors; `body` and `desc` become extra sources. Position `p` resolves to sink row `p` (`p < S`), record `(p-S)/G` row `(p-S)%G` (`S <= p < B`), or ring row `S + (p-S)%cap` (`p >= B`). |

Descriptor: an I32 input tensor, set per ubatch on the host:

```c
enum ggml_sjkvarn_desc {
    GGML_SJKVARN_DESC_S = 0,      // sink size
    GGML_SJKVARN_DESC_CAP,        // ring capacity
    GGML_SJKVARN_DESC_B,          // sealed end
    GGML_SJKVARN_DESC_N,          // visible end
    GGML_SJKVARN_DESC_QPOS0,      // position of the first query row
    GGML_SJKVARN_DESC_G,
    GGML_SJKVARN_DESC_D,
    GGML_SJKVARN_DESC_RECBYTES,
    GGML_SJKVARN_DESC_HKV,
    GGML_SJKVARN_DESC_B_OLD,      // sealed end before this ubatch: seal [B_OLD, B)
    GGML_SJKVARN_DESC_TYPE_K,     // ring ggml_type
    GGML_SJKVARN_DESC_TYPE_V,
    GGML_SJKVARN_DESC_BODY_TYPE,  // scalar or trellis
    GGML_SJKVARN_DESC_SINK_TYPE,  // F16 sink or inherit the ring type
    GGML_SJKVARN_DESC_N_ENTRIES = 16,
};
```

Because the descriptor is a graph input, the same graph can be reused across decode
steps (and CUDA graph capture keeps working). Only the descriptor values change.

## 3. Backend support

- **CPU.** Each op calls the matching `sj_kvarn.h` routine:
  - seal: `sj_kvarn_seal_group`, with records split across threads
  - attention: the decoders plus online softmax
  - staging: `sj_kvarn_tq6_quantize_row`

  The CPU attention can run in parallel over (query row, head) like the existing
  `flash_attn_ext` CPU path.
- **CUDA.** `supports_op` returns true for `TQ6_0` set_rows, the seal and the region-aware
  FA. The kernels in `sj_kvarn_cuda.cuh` produce the right bytes and values but are slow:
  one thread per record or per row. A production FA reads the 4-bit fragment-order
  payload straight into int8 MMA fragments. That order exists for this purpose.
- **Other backends.** `supports_op` returns false, and the scheduler falls back to the CPU.

## 4. `llama-kv-cache`

1. Config:
   - `llama_sj_kvarn_config {enabled, bits_k, bits_v, body, sink = 128, tail = 4096, tail_max = 8192, flush_chunk = 0, iters = 16}`
   - CLI flags: `--sjkvarn-bits K/V`, `--sjkvarn-tail N`, `--sjkvarn-tail-max N` (0 = fixed tail)
2. Allocation per layer:
   - `k`/`v` tensors of `S + cap` rows in `TQ6_0`, where `cap = sj_kvarn_ring_capacity(tail, tail_max, 128, n_ubatch)`
   - an fp16 sink area of `S` rows
   - the I8 body pool for `ceil((n_ctx - S)/G)` groups
3. Position mapping: `cpy_k`/`cpy_v` row indices are the ring slots
   `S + (p - S) % cap`, or `p` for `p < S`.
4. Before each ubatch: `sj_kvarn_policy_begin_ubatch(&pol, pos0, n_tokens)`.
   - If it returns `B_pending > B`, add one `seal_dyn` node per layer ahead of that
     layer's K/V write. It is safe because it reads only positions below `pos0`.
   - After the graph runs, call `sj_kvarn_policy_commit`.
5. Idle: an optional server hook calls `sj_kvarn_policy_idle` and runs a seal-only graph.
6. Unsupported for now, rejected at init with a clear message:
   - sequence copy or removal in the middle of the cache (`seq_rm` other than truncating
     back into the tail)
   - multiple sequences per cache
   - context shift

   Truncating back into the tail is cheap, because it only moves `N`. Truncating below
   `B` needs a re-seal, or a reset of that sequence.

## 5. Tests for the PR

- `test-backend-ops` cases:
  - `TQ6_0` set_rows (against `sj_kvarn_tq6_quantize_row`)
  - seal_dyn (bytes compared against `sj_kvarn_seal_group`, using `tests/expected_vectors.txt` as fixed inputs and hashes)
  - region-aware FA (against `sj_kvarn_layer_attend_row`, tolerance 1e-5 relative)
- A perplexity or KL comparison against an fp16 cache at a context past `sink + tail_max`,
  so that the body is actually used.
