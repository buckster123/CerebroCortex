# CLAUDE.md - CerebroCortex

## Project Overview

CerebroCortex is a brain-analogous AI memory system. It extends Neo-Cortex with:
- **6 memory modalities**: episodic, semantic, procedural, affective, prospective, schematic
- **Associative network**: typed/weighted links with spreading activation (igraph)
- **ACT-R + FSRS hybrid**: biologically-inspired activation and spaced-repetition decay
- **Dream Engine**: LLM-powered offline consolidation (SWS replay, pattern extraction, schema formation, pruning, REM recombination)
- **Triple-backend storage**: ChromaDB (vectors) + SQLite (graph persistence) + igraph (in-memory traversal)

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
# Tests
PYTHONPATH=src pytest tests/ -v

# Quick import check
PYTHONPATH=src python3 -c "from cerebro.types import MemoryType; print('OK')"
```

## Data Location

- SQLite graph: `data/cerebro.db`
- ChromaDB vectors: `data/chroma/`
- Exports: `data/exports/`

## Key Design Decisions

1. **igraph + SQLite** for graph (not Neo4j - too heavy for RPi5)
2. **3 ChromaDB collections** (not 6 like Neo-Cortex - type is metadata filter)
3. **ACT-R power-law decay** B(t) = ln(Σ t_k^{-d}) instead of exponential
4. **4-weight combined scoring**: 35% vector + 30% activation + 20% retrievability + 15% salience
5. **Configurable LLM**: Claude API primary, Ollama fallback for Dream Engine

## Plan Reference

Full implementation plan: `~/.claude/plans/jiggly-sleeping-octopus.md`
