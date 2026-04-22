# Changelog

All notable changes to CerebroCortex are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-04-22

### Added

- **OpenAI-compatible embeddings backend** (`cerebro.storage.embeddings.OpenAICompatEmbeddings`). Parallels the existing `OpenAICompatProvider` for LLM calls — POSTs to `/v1/embeddings` on any configurable base URL. Works with [ryzenai-serve](https://github.com/buckster123/ryzenai-serve) v0.2.0+, LMStudio, vLLM, or any OpenAI-API-shaped embedder. `get_embedding_function("auto")` now prefers `openai_compat` when `OPENAI_COMPAT_EMBEDDING_BASE_URL` is set and reachable, then falls back to sentence-transformers → ollama → hash as before. Enables fully-local dream cycles: NPU LLM + CPU/NPU embedder, no cloud.
  - New `settings.json` keys: `llm.openai_compat_embedding_base_url`, `llm.openai_compat_embedding_model`.
  - Uses `from cerebro import config as _config` pattern (not `from cerebro.config import X`) so `settings.load_on_startup()` overrides actually reach the embedder.

- **Embedder fingerprinting for ChromaDB collections** (`cerebro.storage.embedder_fingerprint`). Closes a silent-data-corruption foot-gun: swapping the embedder model or dim while a collection already holds vectors from a previous model used to fail silently, leaving cosine distance relationships meaningless.
  - Each collection is stamped with `cc_embedder_name` + `cc_embedder_dim` + `cc_embedder_schema` at creation time.
  - Every re-open of an existing collection compares stored-vs-current fingerprint; a loud, actionable WARNING fires once per (collection, mismatch-pair) per process, with three remediation paths.
  - Legacy pre-v0.3.0 collections (no fingerprint) log an INFO hint, not a warning.

- **`cerebro doctor audit` CLI command.** Ops-friendly view of fingerprint health across all collections. `--fix-fingerprint` stamps the current embedder's fingerprint onto legacy or stale collections without re-embedding (use when you've just run `scripts/reembed_collections.py` out-of-band, or on a trusted corpus you're accepting as-is). `--json` for programmatic consumption.

- **`scripts/reembed_collections.py`** — idempotent corpus re-embed when changing embedder. Reads every document, re-embeds via the currently-configured embedder, calls `collection.update(ids=..., embeddings=...)`, and stamps the new fingerprint. Safe to re-run.

### Fixed

- **CLI ignored `~/.cerebro-cortex/settings.json`** (commit 98156a1). `cerebro dream run` and other CLI commands constructed `CerebroCortex()` directly without calling `settings.load_on_startup()`, so JSON overrides silently had no effect — the dream cycle would try to contact the hardcoded default LLM endpoint or fall through to anthropic demanding an API key. CLI commands now call `load_on_startup()` at the top of each command just like the MCP server path does. Same fix applied to the new `doctor audit` command.

- **Schema init migration ordering** (commit c998cb0). The `idx_dream_cycle` index was being created against a column that didn't exist yet on fresh DBs; moved the index creation into the v4 migration block so it lands after the column add.

### Internal

- 16 new unit tests for embedder fingerprinting (`tests/test_storage/test_embedder_fingerprint.py`) covering dim probe fallback, schema drift detection, once-per-pair warning deduplication, and the `strip_immutable_for_modify` helper used to work around ChromaDB 1.x rejecting `hnsw:*` keys on `collection.modify()`.

### Notes on BGE-on-NPU

The original Phase 3 roadmap target ("BGE embedder on the Ryzen AI NPU") was investigated deeply during this cycle and found structurally blocked on Linux SDK 1.7.1: AMD's SDK ships ~48 DPU-kernel xclbins referenced in `vaip_config.json` but installs only 13 `.xclbin` files, none of them the Strix/Krackan DPU overlays needed for BERT-class compilation. The LLM flow dodges this because `onnxruntime_genai_ryzenai` uses a different runtime (IRON/dynamic-dispatch) than the Vitis-AI-EP + DPU path. This is also why AMD ships their NPU-Nomic reference repo as a pre-compiled VAIML cache rather than a compileable recipe. Escape paths (AMD approving the Nomic gate, SDK 1.8 shipping the missing xclbins, or a ROCm iGPU embedder via Radeon 840M) are tracked in intention `mem_e26dfbbc9f5b`. Until then, the v0.2.0-introduced CPU-INT8 BGE embedder (~1.1ms/sentence, 0.996+ cosine fidelity vs FP32) remains the shipping default — fully local, fully operational.

## [0.2.0] — 2026-04-21

- igraph multi-process sync via `PRAGMA data_version`
- Hermes Agent integration report + skill documentation
- Initial release as pip-installable `cerebro-cortex` on PyPI

(Earlier history not retrospectively cataloged — see `git log` before 2026-04-21.)

[0.3.0]: https://github.com/buckster123/CerebroCortex/releases/tag/v0.3.0
[0.2.0]: https://github.com/buckster123/CerebroCortex/releases/tag/v0.2.0
