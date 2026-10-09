# SJ-KVaRN

SJ-KVaRN is a compressed KV-cache format for transformer inference. The name stands for
**Staged, Journaled KVarN**. In this document, **KVarN** means the original method and its
official repository ([huawei-csl/KVarN](https://github.com/huawei-csl/KVarN)), and
**SJ-KVaRN** means the format in this repository.

- **Staged.** Each new K/V row is first written to a staging ring in `tq6_0`, a 6-bit
  format with one fp16 norm per 128 values (6.125 bits per element). For recent tokens,
  attention reads straight from this ring.
- **Journaled.** The ring works like a write-ahead log. Recent tokens stay in this
  near-exact tail and are committed ("sealed") into the compact body in batches of whole
  128-token groups. By default the tail grows from 4096 to 8192 positions and is then
  committed back down to 4096 in a single batch. Sealing therefore happens rarely and in
  large chunks, and the newest 4096 to 8192 tokens are never in the low-bit body.

The sealed body uses the KVarN method:

1. Rotate each head with a Hadamard transform.
2. Balance the variance of each (group, head) tile in the log domain.
3. Quantize each row asymmetrically.
4. Absorb the scales into the record.

At 3 bits and below, SJ-KVaRN codes the body with a trellis instead of plain rounding
(see [Scalar and trellis bodies](#scalar-and-trellis-bodies)).

This repository is an independent implementation and contains no code from KVarN.

What the repository provides:

| File | Contents |
|---|---|
| `sj_kvarn.h` | Single-header C99 library (stb style): formats, rotation, seal (encode), decode, `tq6_0` staging, the adaptive-tail policy, a reference per-layer cache and a reference attention loop. Needs only `libm`. |
| `sj_kvarn_cuda.cuh` | CUDA reference kernels for seal, decode, staging and attention. They compile the same codec source for the device and produce the same bytes. They are written for correctness, not speed. |
| `tests/` | Bit-exactness vectors, policy, KL and fp16 tests (plain C99), a CUDA device-vs-host test, and harnesses that compare against a llama.cpp tree carrying the SJ-KVaRN cache (llamAmpere). |
| `INTEGRATION_LLAMACPP.md` | Sketch of the ggml type and op registration for a llama.cpp pull request. |

## Which version this is

This is the general version of SJ-KVaRN, meant for other engines and for upstream use. It
stores exactly the same bytes as the SJ-KVaRN cache in the llamAmpere fork, so it gives the
full quality reported below. The fork has extra speed work that is not in this repository:
tuned GPU seal and attention kernels, and the Hadamard rotation folded into the attention
kernel and into the cache write. This version keeps the rotation as a separate step. An
engine built from it will be somewhat slower than the fork build; how much depends on the
kernels the integrator writes. What has been measured:

- **Folding the rotation in** was speed-neutral on its own in the fork. Fused Q/output
  rotation: +0.03% / +0.04% per decode round at 4/4 (inside the noise), +0.2% to +0.5%
  (slower) at 3/3 trellis. Fused K/V rotation in the cache write: +0.20% (4/4) and +0.01%
  (3/2 trellis) per decode round, inside the card's run-to-run drift. Both were bit-exact.
- **The reference CUDA kernels in this repository** are far slower than the fork's tuned
  kernels (RTX 3090 Ti, D = 256, 4 KV heads): sealing a 4096-token commit takes 953 ms vs
  10.4 ms at 4/4 and 3,399 ms vs 16.9 ms at 3/2 trellis; one attention call at 100K
  context takes 16 to 26 s vs 0.22 to 0.87 ms. The reference kernels use one thread per
  record or per row and exist to check the format on the device. Both sides wrote
  identical record bytes.

## Results

All results below are on Qwen3.8 27B (4-bit weights, 16 full-attention layers, 4 KV heads
of 256 dims), on an RTX 3090 Ti. KL is the mean full-vocabulary KL divergence (nats) of
the next-token distribution, scored only on model-generated tokens.

### KL against paper-faithful KVarN at 4/4, 3/3 and 3/2

"Paper-faithful KVarN" is a public third-party llama.cpp port of KVarN (BeeLlama v0.4.6),
run two ways: with the 1,024-token fp16 exact tail its README recommends, and with no
extra tail ("tail 0", which still keeps a 128-token exact sink and a 128-token exact
suffix; this is the closest match to the KVarN reference setup).

**Reference.** Each engine is scored against its own 16-bit-cache run on the same
positions. The two engines' 16-bit runs differ from each other by 0.000847 nats on
average, which is about the size of the 4/4 error being measured, so a shared reference
would hide the result.

**Data.** Nine real histories (3 agentic, 3 retrieval, 3 coding) of 9.4K to 15K tokens,
replayed at decode width 4. 964 scored positions, all model-generated tokens with a sealed
body behind them. Brackets are paired-position bootstrap 95% CIs; "task CI" resamples the
nine histories.

**Current build** (llamAmpere DEF1006, c57ebd39a, 2026-10-07; default flags: `tq6_0`
staging, fp16 sink 128, adaptive tail 4096/8192, 4/4 scalar body, 3/3 and 3/2 trellis
body on the token axis). The stock port has no trellis, so its 3/3 and 3/2 bodies are
scalar; these rows compare default configurations.

| Bits K/V | SJ-KVaRN | KVarN, 1,024 tail | SJ-KVaRN lower by | Task CI | KVarN, tail 0 | SJ-KVaRN lower by | Task CI | Histories won |
|---|---|---|---|---|---|---|---|---|
| 4/4 | 0.000979 | 0.001566 | **37.5%** (27.0–46.6) | 30.5–47.9 | 0.001944 | **49.7%** (39.1–58.4) | 42.2–56.9 | 9/9, 9/9 |
| 3/3 | 0.001542 | 0.004338 | **64.4%** (55.5–71.7) | 60.7–71.3 | 0.005444 | **71.7%** (65.0–77.1) | 67.6–74.8 | 9/9, 9/9 |
| 3/2 | 0.002241 | 0.009450 | **76.3%** (70.6–81.0) | 74.0–80.8 | 0.012030 | **81.4%** (77.5–84.8) | 78.6–86.1 | 9/9, 9/9 |

Bits per value actually allocated, averaged over the scored positions:

| Bits K/V | SJ-KVaRN | KVarN, 1,024 tail | KVarN, tail 0 |
|---|---|---|---|
| 4/4 | 5.42 | 5.76 | 4.65 |
| 3/3 | 4.98 | 4.87 | 3.67 |
| 3/2 | 4.76 | 4.43 | 3.19 |

The most likely next token changed at 8 of 964 positions for SJ-KVaRN 3/2, against 24
(1,024 tail) and 29 (tail 0) for KVarN; at 3/3, 6 against 21 and 19.

**Same body on both sides** (scalar at 4/4 and 3/3; earlier llamAmpere build, 2026-09-22):

| Bits K/V | SJ-KVaRN | KVarN, 1,024 tail | SJ-KVaRN lower by | Task CI | KVarN, tail 0 | SJ-KVaRN lower by | Task CI |
|---|---|---|---|---|---|---|---|
| 4/4 scalar | 0.000987 | 0.001566 | 37.0% (26.7–46.5) | 30.5–46.0 | 0.001944 | 49.2% (38.4–59.0) | 41.4–54.3 |
| 3/3 scalar | 0.002273 | 0.004338 | 47.6% (36.8–56.4) | 41.2–56.9 | 0.005444 | 58.3% (48.4–66.7) | 45.6–66.3 |

On that build SJ-KVaRN allocated 5.32 (4/4) and 4.82 (3/3) bits per value. All nine
histories favoured SJ-KVaRN in every row.

**Closer to equal bytes** (current build, same positions):

| Comparison | Bits per value | SJ-KVaRN lower by | Task CI | Histories won |
|---|---|---|---|---|
| SJ-KVaRN 3/3t vs KVarN 4/4, tail 0 | 4.98 vs 4.65 | 20.7% (3.3–35.1) | 11.2–34.0 | 9/9 |
| SJ-KVaRN 3/2t vs KVarN 4/4, tail 0 | 4.76 vs 4.65 | −15.3%, not significant (−42.4 to 8.5) | −34.7 to 8.2 | 4/9 |
| SJ-KVaRN 4/4 vs KVarN 4/4, 4,096 fp16 tail | 5.42 vs 9.07 | 20.2% (5.9–32.4) | 12.7–29.8 | 9/9 |
| SJ-KVaRN 4/4 vs KVarN 4/4, 8,192 fp16 tail | 5.42 vs 13.49 | −3.9%, not significant (−23.8 to 12.8) | −18.3 to 9.2 | 5/9 |

**Caveats.**

- These histories are short (9.4K to 15K tokens), so SJ-KVaRN's 4096 to 8192-token
  staging tail is a large share of the cache. The main table compares default
  configurations, not equal memory. At a 110,592-token context the same SJ-KVaRN
  configurations allocate 4.72 (4/4), 3.76 (3/3t) and 3.27 (3/2t) bits per value.
- Paper-faithful KVarN has not yet been scored at ~100K tokens (the run is in progress).
- A 200K-token needle study (answer-token KL, 6 documents × 8 needles, earlier
  channel-axis trellis) found SJ-KVaRN 4/4 significantly lower than stock at 3 of 8
  look-back distances and never significantly higher; for 3/3t and 3/2t against the
  matching stock configurations, every interval included 1. That study is underpowered
  per distance.

### KL against other KV-cache formats at ~100K tokens

One agentic trajectory of 105,699 tokens (seed 7300), 6,250 scored generated-token
positions (stride 16), every format on the same build (llamAmpere DEF1006) and scored
against the same fp16-cache reference. Bits per value for SJ-KVaRN are the allocation at
the last token; the average over the scored positions is in brackets.

| K / V cache | Bits per value | Mean KL | Same top token |
|---|---|---|---|
| q8_0 / turbo4 | 6.31 | 0.000732 | 99.44% |
| **SJ-KVaRN 3/3 trellis** | **3.49** (3.85) | **0.001145** | 99.20% |
| **SJ-KVaRN 3/2 trellis** | **3.03** (3.44) | **0.002427** | 98.85% |
| q4_0 / q4_0 | 4.50 | 0.002544 | 98.88% |
| q8_0 / turbo3 | 5.81 | 0.002683 | 98.70% |
| turbo3 / turbo3 | 3.12 | 0.007333 | 97.76% |

Paired comparisons (block bootstrap 95% CI over positions):

| Comparison | Trajectories | SJ-KVaRN lower by | 95% CI |
|---|---|---|---|
| 3/3t vs q8_0 / turbo3 | 1 | 57.3% | 50.9–63.0 |
| 3/3t vs q4_0 / q4_0 | 2 | 53.6% | 49.1–57.6 |
| 3/3t vs turbo3 / turbo3 | 2 | 84.4% | 82.7–85.9 |
| 3/2t vs turbo3 / turbo3 | 2 | 68.6% | 65.1–72.0 |
| 3/2t vs q4_0 / q4_0 | 2 | 6.6%, not significant | −3.9 to 16.2 |
| 3/2t vs q8_0 / turbo3 | 1 | 9.5%, not significant | −6.0 to 22.7 |
| 3/3t vs q8_0 / turbo4 | 1 | −56.3% (higher) | −79.5 to −35.7 |
| 3/2t vs q8_0 / turbo4 | 1 | −231.4% (higher) | −292.0 to −179.1 |

Reading it: SJ-KVaRN 3/3t has less than half the KL of q4_0 / q4_0 and of the 5.81-bit
q8_0 / turbo3 cache while storing fewer bits than either. SJ-KVaRN 3/2t, at about 3 bits
per value, is level with the 4.5-bit q4_0 and has 69% less KL than turbo3 at the same size.
Where SJ-KVaRN loses: q8_0 / turbo4, at 1.8 times the bits of 3/3t, has lower KL than both
SJ-KVaRN configurations. The two-trajectory pairs add a coding trajectory (103,454 tokens);
the q8_0 rows have one trajectory so far. A matched 4/4 comparison on this build is still
running.

## Scalar and trellis bodies

Both body types store the same number of bits per value; they differ in how the codes are
chosen.

- **Scalar.** Each number is rounded on its own to the nearest of 2^b levels (16 levels at
  4 bits, 8 at 3 bits). Simple, fast to encode, and each rounding error is independent.
- **Trellis.** The codes for a run of numbers are chosen together. Each code, together with
  the previous few code bits, selects a value from a small trained codebook, so the codes
  move through a small state machine (64 states). A Viterbi search finds the sequence of
  codes whose decoded values follow the data most closely. The same bits can then land
  closer to the real values. Here the run is the 128 tokens of one channel in a group
  (the "token axis").

Measured KL ratios, trellis ÷ scalar at the same bits (lower is better):

| Comparison | Data | KL ratio |
|---|---|---|
| 3/3 token-axis trellis ÷ 3/3 scalar | 2 trajectories of ~100K tokens | 0.646 |
| 3/2 token-axis trellis ÷ 3/2 scalar | 2 trajectories of ~100K tokens | 0.469 |
| 3/3 token-axis ÷ 3/3 channel-axis trellis | 6 trajectories of ~100K tokens | 0.769 (CI 0.725–0.819) |
| 3/2 token-axis ÷ 3/2 channel-axis trellis | 6 trajectories of ~100K tokens | 0.772 (CI 0.730–0.823) |
| 3/3 channel-axis trellis ÷ 3/3 scalar | 3 trajectories of 32K tokens | 0.873 |
| 3/2 channel-axis trellis ÷ 3/2 scalar | 3 trajectories of 32K tokens | 0.718 |
| 2/2 channel-axis trellis ÷ 2/2 scalar | 3 trajectories of 32K tokens | 0.804 |

This library implements the token-axis trellis, the current default; the channel-axis
variant is not included. The first two rows have no confidence intervals.

The cost is in encoding. The Viterbi search makes sealing slower: in the fork's GPU kernel
a 4096-token commit takes 16.9 ms at 3/2 trellis against 10.4 ms at 4/4 scalar, and the
single-thread reference seal in this library takes 88.5 to 108 ms per record against
19.8 ms (see [Limitations](#limitations)). Decoding reads one trained codebook value per
code instead of computing a level; this library's reference decoder takes 7.6 to 8.8 us
per K+V row across body types. In the fork's tuned attention kernels at 100K context, a
3/2 trellis attention call took 0.86 to 0.87 ms against 0.22 to 0.43 ms for 4/4 scalar
(different kernels, decode widths 1 to 5). The adaptive tail seals rarely and in large
batches, so the encode cost comes in occasional bursts. This is why `BODY_AUTO`
picks the trellis at 3 bits and below, where it lowers KL the most, and scalar at 4/4.

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
llamAmpere tree has no scalar kernels for them.

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

## Tests and bit-exactness

| Command | What it checks |
|---|---|
| `make check` | C99 build with `-std=c99 -Wall -Wextra -Wpedantic -Werror`. Also runs fp16 conversion, 21 bit-exact vectors (D = 128 and 256, 10 body configs, plus `tq6_0`), 6.7 million policy steps and the KL sanity test. |
| `tests/build_tree_harness.sh <tree> <build>` then `tests/tree_harness [--kvd FILE]` | Runs a llama.cpp tree's own ggml CPU graph next to this library through the adaptive-tail schedule with a wrapping ring, and compares every byte. |
| `tests/build_tree_harness.sh <tree> <build> cuda` then `tests/tree_cuda_harness` | Runs the tree's own CUDA staging and seal kernels and compares the bytes with this library. |
| `tests/build_cuda_test.sh` then `tests/test_cuda` | Compares device and host on seal records, decode, staging and attention. |

Results against the reference tree (llamAmpere's SJ-KVaRN cache; CPU graph, D = 256, 2 KV heads, tail 256 to 512 in
the harness):

- **End-to-end harness, 10 configs** (4/4s, 3/3t, 3/2t, 2/2t, 4/3s, 3/3s, 4/2s, 2/4s,
  3/2s, 2/2s). Bit-exact for records, staging bytes, sink, rotation, decode and the
  policy schedule.
- **Attention output.** Matches within 2.8e-7 relative. The tree's CPU build contracts
  multiply-adds into FMA, but the format bytes are not affected.
- **Real K/V from a model.** On 48 groups of agentic-task activations, 0 of 48 records
  differ in any of the 10 configs.

- **CUDA, against this library's CPU code.** `tests/test_cuda` passes on an RTX 3090 Ti (sm_86,
  CUDA 12.4). In 6 configs, 8 of 8 sealed records match byte for byte and 0 of 2,048 decoded rows
  differ. The tq6_0 staging rows (2,400) match byte for byte. The attention output matches within
  2.95e-7 relative.
- **CUDA, against the tree's CUDA kernels.** `tests/tree_cuda_harness` runs the tree's own staging and
  seal kernels. Given the same staged input, 0 of 8 records differ in all 10 configs, and 0 of 1,024
  sink rows differ. The tree's CUDA staging differs from this library in 2 of 4,336 rows, by one unit
  in the last place of the fp16 row norm. The tree builds its CUDA code with `-use_fast_math` and sums
  the norm in warp order, while its CPU code (and this library) sums in sequence. This library matches
  the tree's CPU staging exactly. The harness therefore copies those 2 rows across before checking the
  seal, and still reports them.

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

This library matches the reference tree (the SJ-KVaRN cache in the llamAmpere fork of
llama.cpp) byte for byte in every configuration listed above. Compared with that tree it
differs in the following ways.

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

- Only the configs listed above were run on the device, on one GPU (sm_86).

**Compared with KVarN itself:**

- The staging ring, the adaptive tail, the fp16 sink, the trellis bodies and the fragment
  payload order are additions on top of the KVarN method.
- The record layout is not byte-compatible with the KVarN repository's storage.

## Limitations

- **Body variants.** This general version leaves out the 4-bit trellis, the channel-axis trellis
  and the tiered bodies.
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
- **Portability over speed.** The reference codec puts portability ahead of speed. Integrators' fused
  kernels can be much faster.
- **CUDA kernels.** They use one thread per record or per row. They exist to check the
  format on the device, not to run fast.
- **Sequences.** One sequence per cache: the policy tracks a single position stream.
  Multi-sequence batching needs one policy and one set of ring and body buffers per
  sequence.
- **Model coverage.** The trellis codebooks are fixed trained tables. Any model decodes
  correctly, but the quality results above were measured on Qwen3.8 27B only.
- **Group size.** `G` is fixed at 128 for trellis bodies. Head dimension must be a multiple
  of 128.

## Prior work and credits

- **KVarN**, the method the sealed body implements: L. K. Muller, P. Bich, C. Boretti,
  H.-M. Chang, J. Zhuang, L. Cavigelli, "KVarN: Variance-Normalized KV-Cache Quantization
  Mitigates Error Accumulation in Reasoning Tasks", 2026,
  [arXiv:2606.03458](https://arxiv.org/abs/2606.03458); official code
  [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN) (Apache-2.0). The Hadamard
  rotation, the log-domain (Sinkhorn-style) variance balancing, the 128-token groups, the
  scale absorption and the 128-token sink come from KVarN. SJ-KVaRN adds the `tq6_0`
  staging ring, the journaled adaptive tail, the trellis bodies, the tensor-core payload
  order and the record layout.
- **BeeLlama v0.4.6**, the public llama.cpp port of KVarN used as the paper-faithful
  comparison above.
- **Trellis-coded quantization**: M. W. Marcellin and T. R. Fischer, "Trellis coded
  quantization of memoryless and Gauss-Markov sources", IEEE Transactions on
  Communications, 1990. The trellis body, where a code window over the previous code bits
  indexes a codebook and a Viterbi search picks the codes, follows the shift-register
  trellis used by QTIP: A. Tseng, Q. Sun, D. Hou, C. De Sa, "QTIP: Quantization with
  Trellises and Incoherence Processing", NeurIPS 2024,
  [arXiv:2406.11235](https://arxiv.org/abs/2406.11235). turboderp's
  [exllamav3](https://github.com/turboderp-org/exllamav3) (EXL3) is a related
  QTIP-derived weight format. SJ-KVaRN uses its own trained 512-entry (3-bit) and
  256-entry (2-bit) codebooks and contains no code from QTIP or exllamav3.
- **TurboQuant**: A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni, "TurboQuant: Online
  Vector Quantization with Near-optimal Distortion Rate", 2025,
  [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), the rotation plus scalar-codebook
  KV quantization line that the `tq6_0` staging format belongs to.
  [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) is the
  llama.cpp fork that carries the `turbo3` / `turbo4` cache formats compared above; the
  `tq5_0` / `tq6_0` types were developed on top of it.
- **Matrix balancing**: R. Sinkhorn and P. Knopp, "Concerning nonnegative matrices and
  doubly stochastic matrices", Pacific Journal of Mathematics, 1967.

## License

MIT. See `LICENSE` and the end of `sj_kvarn.h`. `NOTICE` credits the KVarN method (huawei-csl, Apache-2.0).

The quantization method of the sealed body comes from KVarN by huawei-csl
(https://github.com/huawei-csl/KVarN), which is licensed under Apache-2.0. This
repository reimplements the method from its description and copies no source code from
KVarN, so it can be distributed under MIT. If you also use code from the KVarN
repository, follow its Apache-2.0 terms for that code. Please credit KVarN when you
publish work that uses this format.
