# Integrating CerebroCortex with OpenClaw

CerebroCortex (CC) is a drop-in memory upgrade for AI agents. It adds decay/promotion, multi-agent convergence, associative graph search, session continuity, and offline memory maintenance — all features that file-based memory systems lack.

CC's MCP server speaks standard MCP over stdio JSON-RPC 2.0. The `openclaw-mcp-adapter` plugin connects to it directly with zero protocol changes.

## Quick Start

### 1. Add to your MCP config

**Claude Code** (`~/.claude.json`):
```json
{
  "mcpServers": {
    "cerebro-cortex": {
      "command": "/path/to/CerebroCore/cerebro-mcp"
    }
  }
}
```

**OpenClaw** (`~/.openclaw/mcp_servers.json`):
```json
{
  "cerebro-cortex": {
    "command": "/path/to/CerebroCore/cerebro-mcp",
    "description": "Long-term memory with search, decay, and consolidation"
  }
}
```

### 2. Verify it works

Once connected, your agent should see tools prefixed with `cerebro_` (or unprefixed, depending on your adapter). Test with:

- **Store**: `remember` with content "Hello from OpenClaw"
- **Search**: `recall` with query "hello"
- **Check**: `memory_health` for a system status report

### 3. Add CLAUDE.md snippet (recommended)

Add this to your agent's system instructions or CLAUDE.md:

```markdown
## Memory System

You have access to a persistent memory system via MCP tools. Key tools:

- `remember` — Save important information (facts, decisions, lessons)
- `recall` — Search memories by meaning (not just keywords)
- `session_save` — Save a session summary before ending
- `session_recall` — Load previous session notes at startup
- `store_intention` — Save TODOs and reminders
- `list_intentions` — Check pending TODOs

At the start of each session, call `session_recall` to orient yourself.
Before ending a session, call `session_save` with a summary.
```

## Configuration

### Agent ID

By default, all memories are stored under agent ID `CLAUDE`. To change this:

**Environment variable** (recommended for multi-agent):
```bash
export CEREBRO_AGENT_ID="my-agent"
```

**Runtime settings** (`data/settings.json`):
```json
{
  "agent": {
    "default_agent_id": "my-agent"
  }
}
```

### API Keys

For the Dream Engine (offline memory maintenance), you need an LLM. Options:

1. **Local LLM** (default): Configure `data/settings.json`:
   ```json
   {
     "llm": {
       "primary_provider": "openai_compat",
       "primary_model": "qwen/qwen3-4b",
       "openai_compat_base_url": "http://localhost:1234"
     }
   }
   ```

2. **Claude API**: Add your key to `data/.env`:
   ```
   ANTHROPIC_API_KEY="sk-ant-..."
   ```

### Runtime Settings

Settings can be changed without restarting the server. The priority order is:
1. `config.py` defaults (lowest)
2. `data/settings.json` overrides
3. `data/.env` for API keys (highest)

## Tool Reference

### Most-Used Tools

| Tool | What it does |
|------|-------------|
| `remember` | Save information to long-term memory |
| `recall` | Search memories by meaning |
| `session_save` | Save session summary for continuity |
| `session_recall` | Load previous session notes |
| `store_intention` | Save a TODO or reminder |
| `list_intentions` | List pending TODOs |
| `ingest_file` | Import a file as searchable memories |

### Memory Management

| Tool | What it does |
|------|-------------|
| `get_memory` | Get a memory by ID |
| `update_memory` | Edit content, tags, or importance |
| `delete_memory` | Remove a memory |
| `share_memory` | Change who can see a memory |
| `associate` | Link two memories together |

### Episodes (Event Sequences)

| Tool | What it does |
|------|-------------|
| `episode_start` | Start recording a sequence of events |
| `episode_add_step` | Add a memory as the next step |
| `episode_end` | Finish recording |
| `list_episodes` | List recent episodes |

### Workflows & Patterns

| Tool | What it does |
|------|-------------|
| `store_procedure` | Save a workflow or how-to guide |
| `create_schema` | Record a pattern from multiple memories |
| `find_relevant_procedures` | Find workflows by topic |

### System

| Tool | What it does |
|------|-------------|
| `memory_health` | System health report |
| `dream_run` | Run offline memory maintenance |
| `register_agent` | Register a new agent |

## Multi-Agent Setup

CerebroCortex supports multiple agents sharing a single memory store with visibility controls.

### Register each agent

```
register_agent(agent_id="researcher", display_name="Researcher", generation=0, lineage="team", specialization="Research and analysis")
register_agent(agent_id="coder", display_name="Coder", generation=0, lineage="team", specialization="Code generation")
```

### Visibility levels

- **shared** (default): All agents can see the memory
- **private**: Only the owning agent can see it
- **thread**: Only agents in the same conversation thread can see it

### Per-agent configuration

Run each agent with a different `CEREBRO_AGENT_ID`:

```bash
# Terminal 1
CEREBRO_AGENT_ID=researcher ./cerebro-mcp

# Terminal 2
CEREBRO_AGENT_ID=coder ./cerebro-mcp
```

## Architecture Notes

- **All data is local**: SQLite for the graph, ChromaDB for vector search, no cloud services required
- **Runs on minimal hardware**: Designed for Raspberry Pi 5 — no Neo4j or heavy dependencies
- **Memories decay naturally**: Unused memories fade over time, frequently-accessed ones get promoted
- **Search is semantic**: Uses sentence embeddings (`all-MiniLM-L6-v2`) so "Python error handling" finds memories about "try/except patterns"
- **Links boost search**: When you find one memory, linked memories are boosted in results automatically

## File Ingestion

Use `ingest_file` to import existing knowledge:

```
ingest_file(file_path="/path/to/notes.md")
ingest_file(file_path="/path/to/data.json", tags=["project", "data"])
ingest_file(file_path="/path/to/script.py", tags=["code", "python"])
```

Supported formats:
- `.md` — Split by headings (## sections) or paragraphs
- `.json` — Array of `{"content": "...", "tags": [...]}` objects
- `.txt`, `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.rb`, `.sh`, etc. — Split by paragraphs, large blocks split at sentence boundaries

## CLAUDE.md Copy-Paste Snippets

### Minimal (just remember/recall)

```markdown
## Memory

Use `remember` to save important information and `recall` to search it later.
```

### Standard (with sessions)

```markdown
## Memory System

You have persistent memory via MCP. At session start, call `session_recall`.
Before ending, call `session_save` with key discoveries and unfinished work.

Core tools: `remember`, `recall`, `store_intention`, `list_intentions`.
```

### Full (all features)

```markdown
## Memory System

You have a persistent memory system with search, decay, and consolidation.

**Every session:**
1. Start: `session_recall` to load context
2. During: `remember` important facts, `store_intention` for TODOs
3. End: `session_save` with summary, discoveries, and next steps

**Search:** `recall` searches by meaning. Use `associate` to link related memories.
**Episodes:** Use `episode_start`/`episode_add_step`/`episode_end` to record event sequences.
**Workflows:** Use `store_procedure` for reusable how-to guides.
**Maintenance:** Run `dream_run` periodically to consolidate and prune memories.
```
