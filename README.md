# SJ-KVaRN

SJ-KVaRN is a compressed KV-cache format for transformer inference. The name stands for
**Staged, Journaled KVarN**:

- **Staged.** Each new K/V row is first written to a staging ring in `tq6_0`, a 6-bit
  format with one fp16 norm per 128 values (6.125 bits per element). For recent tokens,
  attention reads straight from this ring.
- **Journaled.** The ring works like a write-ahead log. Recent tokens stay in this
  near-exact tail and are committed ("sealed") into the compact body in batches of whole
  128-token groups. By default the tail grows from 4096 to 8192 positions and is then
  committed back down to 4096 in a single batch. Sealing therefore happens rarely and in
  large chunks, and the newest 4096 to 8192 tokens are never in the low-bit body.

The sealed body uses the KVarN method from
[huawei-csl/KVarN](https://github.com/huawei-csl/KVarN):

1. Rotate each head with a Hadamard transform.
2. Balance the variance of each (group, head) tile in the log domain.
3. Quantize each row asymmetrically.
4. Absorb the scales into the record.

This repository is an independent implementation and contains no code from KVarN.

What the repository provides:

| File | Contents |
|---|---|
| `sj_kvarn.h` | Single-header C99 library (stb style): formats, rotation, seal (encode), decode, `tq6_0` staging, the adaptive-tail policy, a reference per-layer cache and a reference attention loop. Needs only `libm`. |
| `sj_kvarn_cuda.cuh` | CUDA reference kernels for seal, decode, staging and attention. They compile the same codec source for the device and produce the same bytes. They are written for correctness, not speed. |
| `tests/` | Bit-exactness vectors, policy, KL and fp16 tests (plain C99), a CUDA device-vs-host test, and a harness that compares against a llama.cpp tree carrying the KVarN cache. |
| `INTEGRATION_LLAMACPP.md` | Sketch of the ggml type and op registration for a llama.cpp pull request. |

## Formats and bits

Each sealed record holds one KV head over one group of `G = 128` tokens. The figures
below are for head dimension `D = 256`. Bits per element count both K and V and include
all scales.

| Body | K bits | V bits | Record bytes | Bits / element | Codes |
|---|---|---|---|---|---|
| 4/4 scalar (default) | 4 | 4 | 35,072 | 4.28125 | round to nearest |
| 3/3 trellis | 3 | 3 | 26,880 | 3.28125 | trellis (Viterbi), 64 states |
| 3/2 trellis | 3 | 2 | 22,784 | 2.78125 | trellis |
| 2/2 trellis | 2 | 2 | 18,688 | 2.28125 | trellis |
| scalar 4/3, 3/3, 4/2, 2/4, 3/2, 2/2 | | | | | round to nearest |

`SJKVARN_BODY_AUTO`, the default body setting, chooses trellis for 3/3, 3/2 and 2/2 and
scalar for everything else. The scalar pairs 3/4 and 2/3 also encode and decode, but the
reference tree has no scalar kernels for them.

Two other regions hold positions that are not in the body:

| Region | Positions | Storage | Bits / element |
|---|---|---|---|
| Sink | `[0, 128)` | fp16, K and V | 16 |
| Staging ring (tail) | `[B, N)` | `tq6_0`, 98-byte block per 128 values | 6.125 |
| Body | `[128, B)` | sealed records | see above |

Here `B` is the sealed end and `N` is the number of stored positions. The ring is sized to
`PAD(max(tail, tail_max) + G + 2*n_ubatch, 128)` rows, so that no unsealed row is
overwritten before it is sealed.

### Record layout

All fields are little-endian. All scales are IEEE fp16.

```
K payload   G*D*bits_k/8 bytes
V payload   G*D*bits_v/8 bytes
Kscale[D]   per channel     Kzero[D]  per channel     Ktok[G]  per token
Vch[D]      per channel     Vscale[G] per token       Vzero[G] per token

K[t,d] = (qK[t,d] * Kscale[d] + Kzero[d]) * Ktok[t]
V[t,d] = (qV[t,d] * Vscale[t] + Vzero[t]) * Vch[d]
```

For a scalar body, `q` is the integer code. For a trellis body, `q` is the trained
codebook value selected by the code's window: the current code above the previous six
code bits of the same channel. Values are in the rotated basis. Q is rotated the same
way, and the output is rotated back, since the transform is its own inverse.

**Payload order.** The bit offset of each code is given by `sj_kvarn_code_bit()`.

- **4-bit scalar** payloads are stored in tensor-core fragment order: 16x16 tiles in the
  operand order of an int8 tensor-core MMA. The per-channel K and V scales are permuted to
  match (`sj_kvarn_k_ch_idx`, `sj_kvarn_v_ch_idx`). This makes a fast GPU kernel simpler.
  The reference decoder handles the order for you.
- **Other scalar widths** use a plain token-major bit stream.
- **Trellis** payloads use a channel-major bit stream: the 128-token sequence of each
  channel is contiguous.

**Staging block (`tq6_0`).** The block is `{ fp16 norm; uint8 qs[64]; uint8 qh[32]; }`
over 128 rotated values.

- The 6-bit code `i` has its low nibble in `qs[i/2]` and its top 2 bits in `qh[i/4]`.
- `value = centroid[code] * norm`, where `norm` is `||x||` divided by the norm of the
  reconstructed centroid vector.
- The centroids are the 64 Lloyd-Max levels for N(0, 1/128) built into the header.

**Seal input.** The sealer reads each staged row back from the ring, dequantizes it and
rounds it to fp16. This means the body is encoded from exactly what attention was
already seeing.

### Numerics

Every rounding step that affects the stored bytes goes through an explicit
single-rounding helper. As a result, the record bytes do not depend on `-ffp-contract`,
FMA, or the host versus the device.

Requirements:

- IEEE binary32 and binary64 with round-to-nearest-even
- no flush-to-zero
- `FLT_EVAL_METHOD == 0` (x86-64, AArch64 and CUDA all qualify)
- no `-ffast-math`

## Default configuration

`sj_kvarn_config_default(head_dim, n_head_kv, n_ctx, n_ubatch)` returns:

| Setting | Default | Meaning |
|---|---|---|
| bits K / V | 4 / 4 | scalar body (`BODY_AUTO`) |
| group `G` | 128 | tokens per sealed record |
| sink | 128 | leading positions kept in fp16 |
| staging | `tq6_0` | ring type (fp16 staging is available as an opt-in comparison) |
| tail | 4096 | exact tail left after a commit |
| tail_max | 8192 | tail length that triggers a commit (adaptive tail); 0 = fixed tail |
| flush_chunk | 0 | 0 = commit the whole backlog at once |
| iters | 16 | variance-balancing iterations |

For smaller memory budgets, set `bits_k/bits_v` to 3/3 or 3/2. `BODY_AUTO` then picks
the trellis body.

## Integration in about 10 minutes (llama.cpp-style engine)

This guide uses the reference CPU cache in `sj_kvarn.h`. A GPU engine follows the same
calls with its own kernels. See `sj_kvarn_cuda.cuh` and `INTEGRATION_LLAMACPP.md`.

**1. Add the library.** In one source file:

```c
#define SJ_KVARN_IMPLEMENTATION
#include "sj_kvarn.h"
```

**2. Set up.** Create one cache per attention layer and one policy per sequence:

```c
sj_kvarn_config cfg = sj_kvarn_config_default(head_dim, n_head_kv, n_ctx, n_ubatch);
sj_kvarn_layer layer[n_layer];
for (int il = 0; il < n_layer; ++il) sj_kvarn_layer_init(&layer[il], &cfg);

sj_kvarn_policy pol;
sj_kvarn_policy_init(&pol, cfg.sink, cfg.tail, cfg.tail_max, cfg.group, cfg.flush_chunk);
```

**3. Seal.** At the start of every ubatch whose first position is `pos0`, before any
layer runs:

```c
sj_kvarn_policy_begin_ubatch(&pol, pos0, n_tokens);
if (pol.B_pending > pol.B)
    for (int il = 0; il < n_layer; ++il) sj_kvarn_layer_seal(&layer[il], pol.B, pol.B_pending);
sj_kvarn_policy_commit(&pol);
```

Sealing reads only positions below `pos0`, which earlier ubatches have already stored.
It can therefore run ahead of this ubatch's K/V writes, or on a side stream.

**4. Append.** In each layer, after RoPE, rotate the new K and V heads and store them:

```c
sj_kvarn_rotate_heads(k, n_head_kv, head_dim);       // k, v: [n_head_kv][head_dim] for one position
sj_kvarn_rotate_heads(v, n_head_kv, head_dim);
sj_kvarn_layer_store(&layer[il], pos, k, v);         // sink -> fp16, otherwise -> tq6_0 ring
```

**5. Attend.** Rotate Q the same way. For each query head `hq` (KV head `hq / (n_head/n_head_kv)`):

```c
sj_kvarn_rotate_heads(q, n_head, head_dim);
sj_kvarn_layer_attend_row(&layer[il], pol.B, n_stored, q + hq*head_dim, hq_kv, qpos,
                          1.0f/sqrtf(head_dim), out + hq*head_dim);
sj_kvarn_rotate_heads(out, n_head, head_dim);        // back to the model basis
```

Attention covers three regions:

- `[0, sink)` is read from the fp16 sink.
- `[sink, B)` is decoded from body records.
- `[B, N)` is read from the ring.

**6. Optional idle step.** When the engine has nothing to do, compress a tail that has
grown:

```c
if (sj_kvarn_policy_idle(&pol, n_stored) > pol.B) { /* seal [B, B_pending) on all layers */ sj_kvarn_policy_commit(&pol); }
```

**7. Reset.** On a sequence reset, call `sj_kvarn_policy_reset(&pol)`. The ring and the
body are overwritten as positions come back in.

For a production kernel, replace the reference attention with a flash-attention variant
that reads the three regions in place. The fp32 reference loop is there so you can check
your kernel's output against it. The record decoders `sj_kvarn_decode_k_row` and
`sj_kvarn_decode_v_row`, and the score and accumulate helpers `sj_kvarn_record_scores`
and `sj_kvarn_record_accum_v`, give you the exact values a kernel must reproduce.

## Tests and results

| Command | What it checks |
|---|---|
| `make check` | C99 build with `-std=c99 -Wall -Wextra -Wpedantic -Werror`. Also runs fp16 conversion, 21 bit-exact vectors (D = 128 and 256, 10 body configs, plus `tq6_0`), 6.7 million policy steps and the KL sanity test. |
| `tests/build_tree_harness.sh <tree> <build>` then `tests/tree_harness [--kvd FILE]` | Runs a llama.cpp tree's own ggml CPU graph next to this library through the adaptive-tail schedule with a wrapping ring, and compares every byte. |
| `tests/build_cuda_test.sh` then `tests/test_cuda` | Compares device and host on seal records, decode, staging and attention. |

Results against the reference tree (CPU graph, D = 256, 2 KV heads, tail 256 to 512 in
the harness):

- **End-to-end harness, 10 configs** (4/4s, 3/3t, 3/2t, 2/2t, 4/3s, 3/3s, 4/2s, 2/4s,
  3/2s, 2/2s). Bit-exact for records, staging bytes, sink, rotation, decode and the
  policy schedule.
- **Attention output.** Matches within 2.8e-7 relative. The tree's CPU build contracts
  multiply-adds into FMA, but the format bytes are not affected.
- **Real K/V from a model.** On 48 groups of agentic-task activations, 0 of 48 records
  differ in any of the 10 configs.

KL sanity check on real activations (one group at a time, synthetic peaked queries;
mean KL in nats / relative error of the attention output):

| Body | Agentic sample | Coding sample |
|---|---|---|
| 4/4 scalar | 1.5e-3 / 0.096 | 8.1e-4 / 0.103 |
| 3/3 trellis | 3.6e-3 / 0.131 | 2.0e-3 / 0.140 |
| 3/2 trellis | 3.6e-3 / 0.246 | 2.0e-3 / 0.264 |
| 3/3 scalar | 6.7e-3 / 0.206 | 3.7e-3 / 0.220 |
| 2/2 scalar | 3.7e-2 / 0.484 | 2.1e-2 / 0.512 |

These are per-group checks of the codec. They are not end-to-end model KL, which also
depends on the tail. With the default tail, the newest 4096 to 8192 tokens are not in
the body.

## Format notes and deviations

This library matches the reference tree byte for byte in every configuration listed
above. Compared with that tree it differs in the following ways.

**Left out on purpose** (speed work specific to one engine, or experimental):

- rotation fused into the Q/O projections and into the cache write
- flash-attention column-packing and prefetch kernels
- a 4/4 trellis body and a 4-bit trellis side
- a channel-axis trellis variant
- a "tiered" body that keeps edge layers at higher precision
- scalar clipping
- alternative trellis refit modes
- `q8_0` staging
- environment-variable overrides

None of these change the bytes of the formats that are included.

**Additions:**

- The scalar pairs 3/4 and 2/3 are accepted.
- Head dimensions other than 256 are accepted. D = 128 seal output is checked against the
  tree's encoder; the full graph path was tested at D = 256.

**Not checked:**

- The tree's own CUDA staging and seal kernels were not compared. The comparison here is
  against its CPU reference.

**Compared with KVarN itself:**

- The staging ring, the adaptive tail, the fp16 sink, the trellis bodies and the fragment
  payload order are additions on top of the KVarN method.
- The record layout is not byte-compatible with the KVarN repository's storage.

## Limitations

- **Seal speed.** The seal is a reference implementation. Single thread on a Ryzen 7
  5800X3D at `-O2`:

  | Body | Time per record (128 tokens x 1 KV head) |
  |---|---|
  | 4/4 scalar | 19.8 ms |
  | 3/2 trellis | 88.5 ms |
  | 3/3 trellis | 108 ms |

  Decoding one K+V row takes 7.6 to 8.8 us. The adaptive tail seals in large batches, so
  this cost comes in occasional bursts. A production engine should seal on the GPU or on
  worker threads.
- **CUDA kernels.** They use one thread per record or per row. They exist to check the
  format on the device, not to run fast.
- **Sequences.** One sequence per cache: the policy tracks a single position stream.
  Multi-sequence batching needs one policy and one set of ring and body buffers per
  sequence.
- **Model coverage.** The trellis codebooks are fixed trained tables. Any model decodes
  correctly, but compression quality was only measured on the activations listed above.
- **Group size.** `G` is fixed at 128 for trellis bodies. Head dimension must be a multiple
  of 128.

## License

MIT. See `LICENSE` and the end of `sj_kvarn.h`.

The quantization method of the sealed body comes from KVarN by huawei-csl
(https://github.com/huawei-csl/KVarN), which is licensed under Apache-2.0. This
repository reimplements the method from its description and copies no source code from
KVarN, so it can be distributed under MIT. If you also use code from the KVarN
repository, follow its Apache-2.0 terms for that code. Please credit KVarN when you
publish work that uses this format.
