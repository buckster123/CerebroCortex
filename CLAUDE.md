# CLAUDE.md - CerebroCortex

## Project Overview

CerebroCortex is a brain-analogous AI memory system running on a Raspberry Pi 5 cluster. Published on PyPI as `cerebro-cortex`. It extends Neo-Cortex with:
- **6 memory modalities**: episodic, semantic, procedural, affective, prospective, schematic
- **Associative network**: typed/weighted links with spreading activation (igraph)
- **ACT-R + FSRS hybrid**: biologically-inspired activation and spaced-repetition decay
- **Dream Engine**: LLM-powered offline consolidation (SWS replay, pattern extraction, schema formation, pruning, REM recombination)
- **Triple-backend storage**: ChromaDB (vectors) + SQLite (graph persistence) + igraph (in-memory traversal)
- **42 MCP tools** including cross-agent messaging (`send_message`/`check_inbox`)
- **Multi-agent**: 3 registered agents (CLAUDE-OPUS, CLAUDE-HAILO, CLAUDE-APEX) sharing one memory store

## Project Structure

```
src/cerebro/
├── types.py, config.py          # Core enums and configuration
├── models/                       # Pydantic data models
├── storage/                      # ChromaDB, SQLite+igraph, embeddings
├── activation/                   # ACT-R, FSRS, spreading activation
├── engines/                      # Brain-region engines (9 engines)
├── cortex.py                     # Main coordinator
├── interfaces/                   # MCP server, REST API, CLI
├── migration/                    # Neo-Cortex import
└── utils/                        # LLM client, timing, serialization
```

## Running

```bash
# MCP server
./cerebro-mcp

# REST API
./cerebro-api

# CLI
./cerebro stats
./cerebro dream --run
```

## Development

```bash
# Tests (use venv — system python lacks deps)
venv/bin/python -m pytest tests/ -v

# Quick import check
PYTHONPATH=src venv/bin/python -c "from cerebro.types import MemoryType; print('OK')"

# Build + check for PyPI
venv/bin/python -m build && venv/bin/python -m twine check dist/*
```

## Data Location

Default: `CEREBRO_DATA_DIR` env var, or `~/.cerebro-cortex/`
Dev mode (via bash scripts): `./data/`

- SQLite graph: `$CEREBRO_DATA_DIR/cerebro.db`
- ChromaDB vectors: `$CEREBRO_DATA_DIR/chroma/`
- Exports: `$CEREBRO_DATA_DIR/exports/`

## Key Design Decisions

1. **igraph + SQLite** for graph (not Neo4j — too heavy for RPi5)
2. **3 ChromaDB collections** (not 6 like Neo-Cortex — type is metadata filter)
   - `cerebro_knowledge`: SEMANTIC + SCHEMATIC
   - `cerebro_sessions`: EPISODIC
   - `cerebro_memories`: PROCEDURAL, PROSPECTIVE, AFFECTIVE
3. **ACT-R power-law decay** B(t) = ln(Σ t_k^{-d}) instead of exponential
4. **4-weight combined scoring**: 35% vector + 30% activation + 20% retrievability + 15% salience
5. **Configurable LLM**: Claude API primary, Ollama fallback for Dream Engine
6. **Cross-agent messaging**: `send_message`/`check_inbox` bypass gating, route to `cerebro_knowledge`, use indexed `recipient` column in SQLite
7. **Schema migration**: Idempotent — `ALTER TABLE` + `try/except`, runs on every startup

## Architecture Notes

- **SQLite is canonical**: igraph rebuilt from SQLite on init. ChromaDB can be backfilled from SQLite.
- **Multi-process igraph sync**: igraph is auto-resynced from SQLite using `PRAGMA data_version` detection. When another process commits a write, the next igraph read operation (`get_neighbors`, `get_degree`, `stats`) detects the version change and rebuilds automatically. Zero polling, zero locks, safe for multi-process access (MCP + API + Dream Engine simultaneously).
- **Pruning is safe**: Only prunes nodes with `degree == 0` (no links) — bridge nodes can't be pruned.
- **Gating bypass**: `send_message()` and `store_intention()` skip the Thalamus gating engine.

## Development Guidelines

- **Always use venv**: `venv/bin/python` — system python lacks pydantic, chromadb, etc.
- **Test before commit**: `venv/bin/python -m pytest tests/ -v` — expect 538+ passing
- **Schema changes**: Bump `SCHEMA_VERSION` in `sqlite_schema.py`, add idempotent migration block
- **New MCP tools**: Add to `TOOL_SCHEMAS` dict + `call_tool()` dispatch + handler function in `mcp_server.py`. Update tool count in `test_mcp_server.py`.
- **New cortex methods**: Follow the pattern of `remember()`/`recall()` — enrichment pipeline for writes, ranking pipeline for reads.
- **ChromaDB metadata**: Keep flat (no nested dicts). Add new fields to `node_to_metadata()` in `chroma_store.py`.

## Roadmap

Active development roadmap with sprint plans: `ROADMAP.md`

## Multi-Agent Setup

Three Pi 5 instances sharing one CerebroCortex:
- **CLAUDE-OPUS**: CerebroCortex dev, runs on this host
- **CLAUDE-HAILO**: Hailo AI accelerator Pi
- **CLAUDE-APEX**: ApexAurum cloud instance

Cross-instance comms via `send_message`/`check_inbox` tools. Messages stored as SEMANTIC (→ `cerebro_knowledge`), tagged `from:{agent}`/`to:{agent}`, indexed by `recipient` column.

## Hardware (incoming)

- Sony IMX500 AI Camera (on-sensor inference, CSI)
- Bosch BME688 Environmental Sensor (temp/humidity/pressure/VOC, I2C)
- MLX90640 Thermal Camera (32x24 IR array, I2C)
