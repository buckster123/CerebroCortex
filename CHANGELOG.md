# Changelog

All notable changes to CerebroCortex are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] — 2026-05-16

### Added

- **MCP tool parity for v0.4.2/v0.5.0 features.** The MCP server (`cerebro-mcp`) now exposes 63 tools (was 56), adding first-class MCP access to features previously available only via REST:
  - `cognitive_bootstrap` — Assemble CCBS cognitive priming modules via MCP. Same query analysis, manual triggers, and token budget enforcement as the REST endpoint.
  - `check_near_duplicates` — Preview potential duplicate memories before ingestion.
  - `activation_heatmap` — Get activation/retrievability data for scatter-plot visualization.
  - `activation_at_risk` — Find fading memories that need revival.
  - `activation_curve` — Project ACT-R decay curve for a specific memory over N days.
  - `query_audit` — Query the audit log with filters (event type, actor, target).
  - `audit_summary` — Summary of audit events grouped by type.

### Changed

- Updated README.md MCP tool count (56 → 63) and added new tool categories to the reference table.
- Updated HERMES_INTEGRATE.md tool count reference (40 → 63).

### Internal

- 7 new MCP handler functions in `mcp_server.py`.
- 14 new test cases for new MCP handlers in `tests/test_interfaces/test_mcp_server.py` (70 total MCP tests, all passing).

## [0.5.0] — 2026-05-16

### Added

- **Cerebro Cognitive Bootstrap System (CCBS)** — Dynamic, query-aware cognitive priming.
  - 12 modular cognitive modules stored as Cerebro memories: soul manifest, core identity, 6 cerebro operational sub-modules (index, ops, session, intentions, agents, meta), and 5 optional modules (technical, analysis, creative, research, communicate).
  - Modules are modular, navigable, and self-referential with associative links between related modules.
  - `scripts/bootstrap_ccbs.py` — one-time ingestion script that reads module markdown files and ingests them as PROCEDURAL/SEMANTIC memories with `ccbs` tags.
  - `POST /bootstrap` REST endpoint — assembles a cognitive prompt block based on query analysis, manual trigger detection, keyword auto-detection, and token budget enforcement.
  - Manual triggers: "Full load", "Solo core", "Debug mode", "Creative mode", "Research mode", "Cerebro mode", "Teach me".
  - Token budget tiers: Minimal (~900 tokens), Standard (~1,600), Full (~4,200). Auto-downgrades if context is tight.
  - Hermes skill: `cerebro-cognitive-bootstrap` installed at `~/.hermes/skills/autonomous-ai-agents/cerebro-cognitive-bootstrap/`.

- **Harness memory migration to Cerebro** — Growing project data moved from static harness memory to living Cerebro memories.
  - Project ecosystem map, integration quick-reference, and session checkpoint protocol now stored as SEMANTIC/PROCEDURAL memories in Cerebro.
  - Harness memory trimmed to static-only: host facts, user communication prefs, stable conventions.
  - Enforces the 90/10 rule: 90% of memory ops go to Cerebro, 10% to harness memory.

### Internal

- 14 new unit tests for CCBS bootstrap assembler (`tests/test_bootstrap/test_ccbs.py`).
- 78 REST endpoints total (was 77).

## [0.4.2] — 2026-05-16

### Added

- **Near-duplicate detection public API** (`cerebro.cortex.find_near_duplicates()`). Search all ChromaDB collections for memories with cosine similarity above a threshold to a given content string. Filters by agent visibility scope.
  - New REST endpoint: `POST /near-duplicates/check` — preview duplicate matches before ingestion.
  - Upload endpoint now returns `memories_skipped` count, showing duplicates/short chunks skipped.
  - Dashboard ingest panel shows skip counts inline.

- **Semantic chunking wired into ingestion pipeline** (`cerebro.ingestion.pipeline`). The existing `SemanticChunker` (sentence-aware, embedding-based topic boundary detection) is now automatically instantiated and passed to `TextAdapter` and `MarkdownAdapter` when `SEMANTIC_CHUNKING_ENABLED=True` (default).
  - `MarkdownAdapter` delegates flat markdown (no headings) to `TextAdapter` with semantic chunking.
  - Configurable via `SEMANTIC_CHUNK_SIZE=512` and `SEMANTIC_CHUNK_OVERLAP=50`.

- **Temporal decay visualization dashboard panel** (`js/panels/decay.js`). New "☠ Decay" panel showing:
  - Scatter plot: memories by age vs. ACT-R activation, colored by layer, sized by salience.
  - "At risk" table: sensory/working memories not accessed in >24h, sorted by lowest FSRS retrievability. Revive button triggers a `recall()` to boost activation.
  - Theoretical decay curve: ACT-R + FSRS projection over 30 days.
  - New REST endpoints: `GET /activation/heatmap`, `GET /activation/at-risk`, `GET /activation/curve/{id}`.

- **Audit logging** (`audit_log` table, schema v7). Security/ops trail for multi-agent setups.
  - Events logged: `access_denial`, `visibility_changed`, `content_edited`, `memory_deleted`, `link_created`.
  - Hooks in `get_memory()`, `delete_memory()`, `update_memory()`, `share_memory()`, `associate()`.
  - Best-effort logging: never blocks main operations.
  - Configurable via `AUDIT_LOGGING_ENABLED=True` (default).
  - REST endpoints: `POST /audit/query` (filterable), `GET /audit/summary` (event-type breakdown).
  - Dashboard "✶ Audit" panel with filterable event list and summary counts.

### Internal

- SQLite schema version bumped from 6 → 7 (`audit_log` table + indexes).
- 25 new tests across dedup, semantic chunking, activation API, and audit hooks.
- 477 total tests passing (156 ingestion/interfaces +  engines + storage).

## [0.4.1] — 2026-05-16

### Added

- **File watcher daemon** (`cerebro.watch`). Watchdog-based recursive directory monitoring for auto-ingestion.
  - `cerebro watch <dirs>` CLI command.
  - API background thread auto-start when `WATCH_ENABLED=True`.
  - Dashboard watcher toggle in header.

- **Dashboard overhaul** (`src/cerebro/web/`). Alchemical theme, modular JS/CSS, 11 panels.
  - 20 new REST endpoints: ingest/upload, trash CRUD, versions, tags, threads, bulk ops, export, watch/status, watch/toggle.

### Internal

- 453 total tests passing.

## [0.4.0] — 2026-05-13

### Added

- **Unified StorageCoordinator** (`cerebro.storage.coordinator.StorageCoordinator`). SQLite is now the single source of truth; ChromaDB is a rebuildable index. All writes route through `StorageCoordinator`, which writes SQLite first and queues ChromaDB backfill on failure. Prevents silent data divergence between graph and vector stores.
  - `backfill_pending()` recovers from transient ChromaDB failures at startup.
  - All 8 dual-write sites migrated: `cortex.py`, `prefrontal.py`, `cerebellum.py`, `neocortex.py`, `text_import.py`, `markdown_import.py`, `json_import.py`, `neo_cortex_import.py`.

- **Unified Ingestion Pipeline** (`cerebro.ingestion`). Pluggable adapter system for importing documents into the full encoding pipeline (no bypassing Thalamus/Semantic/Amygdala).
  - **7 adapters**: `TextAdapter`, `MarkdownAdapter`, `JSONAdapter`, `ImageAdapter`, `PDFAdapter`, `HTMLAdapter`, `CSVAdapter`.
  - Priority ordering: Image > PDF > HTML > CSV > Markdown > JSON > Text.
  - `cortex.bulk_remember()` for efficient batch ingestion through the full pipeline.
  - `IngestionPipeline` auto-detects format from file extension and content sniffing.

- **Vision & Cross-Modal Memory** (`cerebro.storage.vision_embeddings`). Secondary vision embedding function using sentence-transformers CLIP. Separate vector space merged at recall time.
  - `VisionEmbeddingFunction` + `VisionVectorStore` sidecar.
  - `recall(include_vision=True)` merges text and vision results with cross-modal boost scoring.
  - `ImageAdapter` extracts captions (Ollama llava fallback) and optional pytesseract OCR.
  - `PDFAdapter` extracts text + embedded images, links images as `part_of` the document memory.
  - `HTMLAdapter` ingests text + image references.
  - `[vision]` extras in `pyproject.toml`: `pillow`, `pytesseract`, `pymupdf`, `beautifulsoup4`.

- **SQLite FTS5 Full-Text Search** (`cerebro.storage.sqlite_schema`). Keyword fallback when vector search returns empty results.
  - `memory_nodes_fts` virtual table with auto-maintained triggers.
  - `recall()` falls back to FTS5 at 0.5 similarity threshold when ChromaDB returns no matches.

- **Soft Delete & Trash Can** (`cerebro.storage.graph_store`, `cerebro.cortex`).
  - `delete_memory(hard=False)` is the new default — memories go to trash with `deleted_at` timestamp.
  - `list_deleted()` shows trash contents.
  - `restore_memory()` pulls from trash back into active search.
  - `purge_memory()` / `purge_all_deleted(days=N)` for permanent removal.
  - Partial index on `deleted_at IS NOT NULL` for fast trash queries.

- **Memory Versioning** (`cerebro.storage.graph_store`). Audit trail for memory edits.
  - `memory_versions` table snapshots content, tags, salience, and visibility before each content update.
  - `get_memory_versions()` lists history; `get_version()` fetches a specific revision.
  - `restore_version()` rolls back to any previous revision.
  - Snapshots triggered only on content change — metadata-only updates skip versioning for efficiency.

- **Tag Management Engine** (`cerebro.engines.tag_manager`).
  - `list_tags()`, `rename_tag()`, `merge_tags()`, `delete_tag()` — all via client-side JSON manipulation.

- **Bulk Operations** (`cerebro.cortex`).
  - `bulk_update_visibility()` — change visibility for multiple memories at once.
  - `bulk_delete()` — soft-delete multiple memories.
  - `export_memories()` — export to JSON or Markdown.

- **Thread Management** (`cerebro.cortex`).
  - `list_threads()`, `get_thread_memories()`, `prune_thread()`.

- **16 New MCP Tools** (now 56 total).
  - Trash: `list_deleted`, `restore_memory`, `purge_memory`, `purge_all_deleted`
  - Versions: `get_memory_versions`, `restore_version`
  - Tags: `list_tags`, `rename_tag`, `merge_tags`, `delete_tag`
  - Bulk: `bulk_delete`, `export_memories`
  - Threads: `list_threads`, `get_thread_memories`, `prune_thread`
  - Vision: `describe_image`, `search_vision`

- **Pydantic v2 Compliance**. Replaced all 4 deprecated `json_encoders` with `@field_serializer` in `models/memory.py`, `models/link.py`, `models/episode.py`, `models/agent.py`. Zero deprecation warnings.

### Fixed

- **StorageCoordinator.delete_node() not passing `soft` parameter**. The `soft` flag was silently dropped, causing all deletes to be hard deletes even when soft-delete was requested. Now correctly forwarded to `GraphStore.delete_node()`.

- **Schema migration ordering** (v5→v6). `deleted_at` column and `memory_versions` table added safely via `ALTER TABLE` with defaults for legacy rows.

### Internal

- SQLite schema version bumped from 4 → 6.
- 61 new unit tests: 16 ingestion adapters, 14 vision/cross-modal, 13 soft-delete + versioning, 18 CRUD/bulk/thread.
- 447 total tests passing (288 core + 57 migration + 73 interface + 29 new).

[0.4.0]: https://github.com/buckster123/CerebroCortex/releases/tag/v0.4.0
[0.3.0]: https://github.com/buckster123/CerebroCortex/releases/tag/v0.3.0

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
