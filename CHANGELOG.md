# Changelog

## 2.0 (2026-10-09): prompt caching and multiple sequences

Record bytes are unchanged (`SJKVARN_FORMAT_VERSION` stays 1; all 21 bit-exact vectors
pass unchanged). The major version changes because `sj_kvarn_policy_idle` takes a third
argument.

Added:

- `sj_kvarn_policy_truncate`, `sj_kvarn_policy_trunc_floor`, `sj_kvarn_group_floor`,
  `SJKVARN_NO_POS`: cut a sequence at `p`. In the tail, the cut is exact. In the body, the
  cut drops whole sealed groups from `sink + G*floor((p - sink)/G)` on and returns that
  position (at most G - 1 before `p`) as the place to re-prefill from. Nothing is
  re-quantised, and kept records are bit-identical.
- `sj_kvarn_policy_idle(&pol, end, keep_recent)`: the idle step never seals the last
  `max(keep_recent, tail)` positions. **Breaking:** the old two-argument call is
  `sj_kvarn_policy_idle(&pol, end, 0)`.
- `sj_kvarn_seq`: a per-sequence cache (layers, policy, stored end) with `init`, `reset`,
  `begin_ubatch`, `store`, `idle`, `truncate`, `attend_row`.
- `sj_kvarn_pool`: a paged record pool shared by several sequences, with a per-sequence
  block table; `sj_kvarn_layer_record` resolves a record through it.
- Versioned state save/restore (`SJKVARN_STATE_VERSION` 1): `sj_kvarn_seq_state_size`,
  `sj_kvarn_state_size_max`, `sj_kvarn_seq_state_write`, `sj_kvarn_state_peek`,
  `sj_kvarn_seq_state_read`, with a config-identity header, an FNV-1a 64 checksum and a
  distinct error code per mismatch (`sj_kvarn_strerror`).
- `sj_kvarn_layer_truncate`: optional scrub of dropped records and rows.
- `tests/test_state.c` and `tests/expected_state.txt`: truncation, state round trips
  (4/4, 3/3, 3/3t, 3/2t, f16 staging), 30 mismatch rejections, idle margin, two
  sequences in a pool vs each alone, golden state hashes. `tests/test_policy.c` now also
  fuzzes truncations and idle margins. `make test` is an alias of `make check`.
- Docs: prompt caching and multi-sequence sections in the README;
  `INTEGRATION_LLAMACPP.md` §4a makes prompt caching a requirement of the llama.cpp
  integration.

These paths are reference implementations for correctness. Speed work (fused
multi-sequence attention over the block table, batched truncation, asynchronous
serialization) is planned for v0.6.

## 1.x

Initial release: formats, seal/decode, `tq6_0` staging, the adaptive-tail policy, the
reference per-layer cache and attention, CUDA reference kernels and tree harnesses.
