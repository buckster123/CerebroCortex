# CerebroCortex v1.1 Roadmap

**Status:** Active — updated as sprints complete
**Version:** v0.1.0 → targeting v1.1.0
**Origin:** Architectural review by Claude-web + codebase evaluation by Claude-Opus

---

## Sprint Overview

| Sprint | Theme | Proposals | Status |
|--------|-------|-----------|--------|
| **1** | Polish & Quick Wins | #9 Reindex, #10 Episode Auto-Close, #8 Contradiction Flagging | `done` |
| **2** | Recall Quality | #1 Explain Mode, #2 Link Decay | `pending` |
| **3** | Dream Hardening | #7 Schema Validation, #3 Dream Checkpointing | `pending` |
| **4** | Reliability | #5 Concurrency Locks, #11 Verify/Repair | `pending` |
| **H** | Hardware Integration | Sensor daemon (BME688, IMX500, MLX90640) | `awaiting hardware` |
| **I** | Infrastructure | MCP-to-REST shim, dedicated cerebro server | `planned` |

---

## Proposals Already Handled

| # | Proposal | Why |
|---|----------|-----|
| 4 | Bridge-Node Protection | Pruning requires `degree == 0` — bridges can't be pruned by design |
| 12 | pgvector Migration Path | `VectorStore` ABC is clean (7 methods). Drop-in replacement ready |
| 6 | Context-Adaptive Weights | Deferred — needs careful design. Weights in config.py are easy to change when ready |

---

## Sprint 1 — Polish & Quick Wins

### 1.1 Embedding Reindex Command (#9)

**Problem:** `backfill_vector_store()` fills missing vectors but can't re-embed with a new model.

**Files:**
- `src/cerebro/cortex.py` — `backfill_vector_store()` (~line 968)
- `src/cerebro/interfaces/cli.py` — add `cerebro backfill --reindex-all`

**Work:**
- [x] Add `reindex_all=False` parameter to `backfill_vector_store()`
- [x] When `reindex_all=True`: delete from ChromaDB, re-embed from SQLite content
- [x] Add `--reindex-all` flag to CLI
- [x] Test: reindex 10 memories, verify similarity scores still work

**Effort:** ~30 lines | **Risk:** Low

---

### 1.2 Episode Auto-Close (#10)

**Problem:** Episodes require manual `episode_end()`. If Claude forgets, episodes dangle forever. In-memory `_active_episodes` dict lost on restart.

**Files:**
- `src/cerebro/engines/hippocampus.py` — episode lifecycle
- `src/cerebro/engines/dream.py` — could auto-close stale episodes before dream cycle
- `src/cerebro/cortex.py` — session_save could trigger auto-close

**Work:**
- [x] Add `EPISODE_AUTO_CLOSE_HOURS = 24` to config.py
- [x] Add `close_stale_episodes(max_age_hours)` to EpisodicEngine
- [x] Call from `session_save()` and Dream Engine phase 1
- [x] Add SQLite query: episodes with no `ended_at` and `started_at` older than threshold
- [x] Log auto-closed episodes for transparency

**Effort:** ~50 lines | **Risk:** Low

---

### 1.3 Contradiction Flagging (#8)

**Problem:** `contradicts` links exist and REM phase creates them, but recall never checks or surfaces them. Two contradicting memories return without any flag.

**Files:**
- `src/cerebro/cortex.py` — post-recall processing (~line 542)
- `src/cerebro/interfaces/mcp_server.py` — add contradiction info to recall response

**Work:**
- [x] After ranking in `recall()`, check top-k results for `contradicts` links between them
- [x] Add `contradictions: dict[str, list[str]]` to recall response metadata
- [x] Surface in MCP recall response: "Note: memories X and Y contradict each other"
- [x] Add `get_contradictions(memory_id)` convenience method

**Effort:** ~50 lines | **Risk:** Low

---

## Sprint 2 — Recall Quality

### 2.1 Recall Explainability (#1)

**Problem:** 9 engines in the pipeline, single float score returned. When a memory doesn't surface, no way to debug why.

**Files:**
- `src/cerebro/cortex.py` — `recall()` method
- `src/cerebro/engines/prefrontal.py` — `rank_results()`
- `src/cerebro/activation/strength.py` — `combined_recall_score()`
- `src/cerebro/interfaces/mcp_server.py` — expose via `explain` parameter

**Work:**
- [ ] Create `RecallExplanation` dataclass with score breakdown
- [ ] Modify `combined_recall_score()` to optionally return component dict
- [ ] Add `explain=False` parameter to `recall()`
- [ ] When explain=True, return `list[tuple[MemoryNode, float, RecallExplanation]]`
- [ ] Expose in MCP `recall` tool and REST `/recall` endpoint

**Effort:** ~60 lines | **Risk:** Low (additive, no breaking changes)

---

### 2.2 Link Decay (#2)

**Problem:** ACT-R/FSRS decay applies to nodes but link weights are static. Old links between unused memories carry full weight forever.

**Files:**
- `src/cerebro/models/link.py` — AssociativeLink (has `last_activated` timestamp)
- `src/cerebro/activation/spreading.py` — link weight usage
- `src/cerebro/config.py` — add `LINK_DECAY_HALFLIFE_DAYS`

**Work:**
- [ ] Add `LINK_DECAY_HALFLIFE_DAYS = 30` to config.py
- [ ] Add `effective_weight(link, now)` function using FSRS-style curve
- [ ] Apply in spreading activation when reading neighbor weights
- [ ] Compute on-the-fly (don't mutate stored weights — Hebbian strengthening handles that)
- [ ] Test: old untouched links should spread less than fresh active ones

**Effort:** ~40 lines | **Risk:** Medium (affects recall quality — needs tuning)

---

## Sprint 3 — Dream Hardening

### 3.1 Schema Validation (#7)

**Problem:** Dream Phase 3 creates schemas that go straight to LONG_TERM with salience 0.9 and stability 30 days. No quality gate.

**Files:**
- `src/cerebro/engines/neocortex.py` — SchemaEngine
- `src/cerebro/engines/dream.py` — Phase 3
- `src/cerebro/types.py` — may need SchemaState enum

**Work:**
- [ ] New schemas start in WORKING layer (not LONG_TERM)
- [ ] Track `support_count` in schema metadata tags
- [ ] Promote to LONG_TERM after: 3+ supporting episodes AND 2+ real recall accesses
- [ ] Add `evaluate_schema_candidates()` to Dream Engine (run in Phase 3)
- [ ] Demote/prune schemas with 0 accesses after 3 dream cycles

**Effort:** ~80 lines | **Risk:** Medium (changes Dream behavior)

---

### 3.2 Dream Checkpointing (#3)

**Problem:** If LLM fails mid-dream, partial results may leave graph inconsistent. No resume from last completed phase.

**Files:**
- `src/cerebro/engines/dream.py` — `run_cycle()`
- `src/cerebro/storage/sqlite_schema.py` — extend `dream_log`

**Work:**
- [ ] Add `cycle_id` and `status` columns to dream_log
- [ ] Before each phase: check if already completed in this cycle
- [ ] After each phase: log completion with cycle_id
- [ ] Add `resume_cycle(cycle_id)` that skips completed phases
- [ ] Add idempotency guards to Phase 2 (dedup procedures) and Phase 3 (dedup schemas)
- [ ] Expose resume via CLI: `cerebro dream --resume`

**Effort:** ~120 lines | **Risk:** High (phase dependencies are complex)

---

## Sprint 4 — Reliability

### 4.1 Concurrency Locks (#5)

**Problem:** No threading locks on igraph mutations. Multiple coroutines could corrupt in-memory graph.

**Files:**
- `src/cerebro/storage/graph_store.py` — all mutation methods
- `src/cerebro/cortex.py` — CerebroCortex singleton

**Work:**
- [ ] Add `threading.RLock()` to GraphStore
- [ ] Wrap all igraph mutation methods (add_node, delete_node, add_link, etc.)
- [ ] Read-only igraph operations (get_neighbors, get_degree) don't need locks
- [ ] Verify SQLite WAL mode is set (it is — `sqlite_schema.py` line 173)
- [ ] Add concurrency test: parallel writes to GraphStore

**Effort:** ~80 lines | **Risk:** Medium (lock contention, must not deadlock)

---

### 4.2 Verify & Repair (#11)

**Problem:** Three data stores can drift. No way to detect or fix inconsistencies.

**Files:**
- `src/cerebro/cortex.py` — add verify/repair methods
- `src/cerebro/interfaces/cli.py` — add commands

**Work:**
- [ ] `verify_consistency()`: compare node IDs across SQLite, ChromaDB, igraph
- [ ] Report: missing from ChromaDB, orphaned in ChromaDB, igraph drift
- [ ] `repair_consistency()`: rebuild ChromaDB from SQLite, rebuild igraph
- [ ] CLI: `cerebro verify`, `cerebro repair --confirm`
- [ ] Add to Dream Engine: light verify check before each cycle

**Effort:** ~120 lines | **Risk:** Medium (repair is destructive — needs confirmation)

---

## Sprint H — Hardware Integration (awaiting delivery)

### Sensor Daemon

**Hardware on order:**
- Sony IMX500 AI Camera (on-sensor inference, CSI)
- Bosch BME688 Environmental Sensor (temp/humidity/pressure/VOC, I2C)
- MLX90640 Thermal Camera (32x24 IR array, I2C)

**Work:**
- [ ] Python daemon: reads sensors on interval (BME688 every 60s, thermal every 300s)
- [ ] Stores readings as episodic memories in CerebroCortex
- [ ] IMX500 object detection events → episodic memories with tags
- [ ] Dream Engine can discover environmental patterns over time
- [ ] Configurable: `SENSOR_POLL_INTERVAL`, `SENSOR_ENABLED` flags

---

## Sprint I — Infrastructure

### MCP-to-REST Shim

**Problem:** Each Pi needs local CerebroCortex code + shared data directory. A thin MCP shim forwarding to the REST API would decouple this.

**Work:**
- [ ] Add `/messages/send` and `/messages/inbox` to REST API
- [ ] Single-file `cerebro-mcp-remote.py` (~100 lines)
- [ ] Deps: `mcp` + `httpx` only (no heavy ML libs)
- [ ] Each Pi configures it in `.claude.json` pointing to central server
- [ ] Test: full tool roundtrip via REST

### Dedicated Cerebro Server
- [ ] Move data directory to dedicated Pi/server with NVMe
- [ ] Run `cerebro-api` as systemd service
- [ ] UPS for power protection
- [ ] All instances connect via REST

---

## Version Plan

| Version | Sprints | Milestone |
|---------|---------|-----------|
| v0.1.0 | Current | PyPI published, 42 MCP tools, cross-agent messaging |
| v0.2.0 | Sprint 1 + 2 | Recall quality + polish |
| v0.3.0 | Sprint 3 + 4 | Reliability + dream hardening |
| v1.0.0 | Sprint H + I | Sensor integration + multi-Pi architecture |
| v1.1.0 | Polish | Production-ready |

---

*Last updated: 2026-02-10 — Sprint 1 complete, Sprint 2 next*
