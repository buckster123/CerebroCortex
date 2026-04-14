# Integrating CerebroCortex with Hermes Agent

CerebroCortex (CC) is a brain-analogous memory system that gives [Hermes Agent](https://github.com/NousResearch/hermes-agent) persistent, associative, emotionally-weighted memory with decay, consolidation, and multi-agent support.

Two integration paths — use one or both:

| Path | What you get | Setup time |
|------|-------------|------------|
| **MCP Server** | 40+ tools as `mcp_cerebro_*` in Hermes | 2 minutes |
| **Memory Provider Plugin** | Prefetch, background sync, session summaries, MEMORY.md mirroring | 5 minutes |

---

## Path A: MCP Server (Recommended Start)

The fastest way to get CerebroCortex into Hermes. All 40 tools appear as `mcp_cerebro_*` tools.

### 1. Install CerebroCortex

```bash
pip install 'cerebro-cortex[all]'
```

Or from source:
```bash
git clone https://github.com/buckster123/CerebroCortex.git
cd CerebroCortex
pip install -e '.[all]'
```

### 2. Add to Hermes config

Edit `~/.hermes/config.yaml`:

```yaml
mcp:
  servers:
    cerebro-cortex:
      command: cerebro-mcp
      # Or full path: /path/to/CerebroCortex/cerebro-mcp
      env:
        CEREBRO_AGENT_ID: "HERMES"  # Optional: identify this agent
```

### 3. Restart Hermes

```bash
hermes
```

You should see CerebroCortex tools loading in the startup output. Verify with:

```
/tools
```

Look for `mcp_cerebro_*` tools (remember, recall, session_save, etc.).

### 4. Test it

In a Hermes session:
```
Store a memory: "CerebroCortex is now connected to Hermes"
Then search for "connected"
```

The agent will use `mcp_cerebro_remember` and `mcp_cerebro_recall` automatically.

### What you get with MCP

- **40+ tools** — remember, recall, episodes, sessions, intentions, schemas, procedures, graph exploration, dream engine, agent messaging
- **Semantic search** — meaning-based recall, not just keywords
- **Memory decay** — unused memories fade, active ones strengthen
- **Associative graph** — linked memories boost each other in search
- **Multi-agent** — shared/private/thread visibility scoping
- **Dream Engine** — offline consolidation, pattern extraction, pruning

---

## Path B: Memory Provider Plugin (Deeper Integration)

The plugin integrates CerebroCortex directly into Hermes's memory pipeline. It provides features that MCP alone can't:

- **Prefetch** — relevant memories injected before each turn (no tool call needed)
- **Background sync** — significant user messages auto-stored as memories
- **Session summaries** — automatic session save on conversation end
- **MEMORY.md mirroring** — writes to Hermes's built-in MEMORY.md are also stored in CerebroCortex
- **Delegation capture** — subagent results stored as memories
- **5 focused tools** — `cc_remember`, `cc_recall`, `cc_todo`, `cc_message`, `cc_health`

### Install

```bash
pip install cerebro-cortex
```

The plugin lives in the Hermes plugin directory. See [PR #7913](https://github.com/NousResearch/hermes-agent/pull/7913) for the full plugin code, or install from the Hermes plugins directory:

```
plugins/memory/cerebrocortex/
├── __init__.py     # Plugin entry point (495 lines)
├── plugin.yaml     # Plugin metadata
└── README.md       # Plugin-specific docs
```

### Configure

The plugin auto-discovers CerebroCortex if installed. No additional config needed beyond ensuring `cerebro-cortex` is pip-installed in the Hermes venv.

Optional environment variables:
```bash
export CEREBRO_AGENT_ID="MY-AGENT"       # Agent identity
export CEREBRO_DATA_DIR="/path/to/data"  # Data location (default: ~/.cerebro-cortex/)
```

### Using both MCP + Plugin

You can run both simultaneously. The MCP server gives you the full 40-tool suite for manual exploration, while the plugin handles automatic prefetch/sync in the background. They share the same SQLite database safely (WAL mode + igraph auto-resync).

---

## Configuration

### Agent ID

Set an agent identity so memories are attributed correctly in multi-agent setups:

```yaml
# In ~/.hermes/config.yaml (MCP path)
mcp:
  servers:
    cerebro-cortex:
      env:
        CEREBRO_AGENT_ID: "HERMES-MAIN"
```

Or as an environment variable:
```bash
export CEREBRO_AGENT_ID="HERMES-MAIN"
```

### Data Directory

Default: `~/.cerebro-cortex/`

Override:
```bash
export CEREBRO_DATA_DIR=/path/to/data
```

Contents:
```
~/.cerebro-cortex/
├── cerebro.db      # SQLite graph database
├── chroma/         # ChromaDB vector store
├── settings.json   # Runtime settings
├── .env            # API keys
└── exports/        # Data exports
```

### Dream Engine (LLM for offline consolidation)

The Dream Engine needs an LLM for pattern extraction and schema formation. Options:

**Local LLM (recommended for Pi 5):**
```json
// ~/.cerebro-cortex/settings.json
{
  "llm": {
    "primary_provider": "openai_compat",
    "primary_model": "qwen/qwen3-4b",
    "openai_compat_base_url": "http://localhost:1234"
  }
}
```

**Claude API (fallback):**
```bash
# ~/.cerebro-cortex/.env
ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Tool Reference

### Most-Used Tools (MCP)

| Hermes Tool | What it does |
|-------------|-------------|
| `mcp_cerebro_remember` | Save information to long-term memory |
| `mcp_cerebro_recall` | Search memories by meaning |
| `mcp_cerebro_session_save` | Save session summary for continuity |
| `mcp_cerebro_session_recall` | Load previous session notes |
| `mcp_cerebro_store_intention` | Save a TODO or reminder |
| `mcp_cerebro_list_intentions` | List pending TODOs |
| `mcp_cerebro_send_message` | Message another agent |
| `mcp_cerebro_check_inbox` | Check for messages |
| `mcp_cerebro_dream_run` | Run offline memory maintenance |
| `mcp_cerebro_ingest_file` | Import a file as searchable memories |

### Plugin Tools (if using memory provider)

| Tool | What it does |
|------|-------------|
| `cc_remember` | Store a memory (with auto-classification) |
| `cc_recall` | Search memories (with activation scoring) |
| `cc_todo` | Store or list TODOs |
| `cc_message` | Send/check inter-agent messages |
| `cc_health` | System health report |

---

## Multi-Agent Setup

CerebroCortex shines in multi-agent Hermes deployments. Each agent gets its own identity with visibility-scoped memory.

### Register agents

```
mcp_cerebro_register_agent(
  agent_id="HERMES-MAIN",
  display_name="Main Agent",
  generation=0,
  lineage="hermes",
  specialization="General assistant"
)
```

### Visibility levels

- **shared** (default) — all agents see it
- **private** — only the owning agent
- **thread** — only agents in the same conversation

### Per-agent MCP instances

Run separate cerebro-mcp instances per agent, each with a unique ID:

```yaml
# ~/.hermes/config.yaml
mcp:
  servers:
    cerebro-cortex:
      command: cerebro-mcp
      env:
        CEREBRO_AGENT_ID: "HERMES-MAIN"
```

### Inter-agent messaging

Agents can send messages to each other through CerebroCortex:

```
mcp_cerebro_send_message(to="HERMES-HAILO", content="Analysis complete, results in memory")
mcp_cerebro_check_inbox(from_agent="HERMES-MAIN")
```

Messages bypass gating (always delivered), are auto-tagged with sender/recipient, and show up in the recipient's inbox.

---

## System Prompt Snippets

Add to your Hermes persona or system prompt for best results:

### Minimal

```
You have persistent memory via CerebroCortex MCP tools (mcp_cerebro_*).
Use mcp_cerebro_remember to save important facts and mcp_cerebro_recall to search them.
```

### Standard (with sessions)

```
You have persistent memory via CerebroCortex. At session start, call mcp_cerebro_session_recall.
Before ending, call mcp_cerebro_session_save with key discoveries and unfinished work.
Core tools: mcp_cerebro_remember, mcp_cerebro_recall, mcp_cerebro_store_intention.
```

### Full (all features)

```
You have a persistent memory system (CerebroCortex) with semantic search, decay, and consolidation.

Every session:
1. Start: mcp_cerebro_session_recall to load context
2. During: mcp_cerebro_remember for important facts, mcp_cerebro_store_intention for TODOs
3. End: mcp_cerebro_session_save with summary, discoveries, and next steps

Search: mcp_cerebro_recall searches by meaning. Use mcp_cerebro_associate to link related memories.
Episodes: Use episode_start/add_step/end to record event sequences.
Maintenance: Run mcp_cerebro_dream_run periodically to consolidate and prune memories.
```

---

## Architecture Notes

- **All data is local** — SQLite + ChromaDB + igraph, no cloud services required
- **Runs on Raspberry Pi 5** — designed for minimal hardware (4GB+ RAM)
- **Multi-process safe** — MCP server, REST API, and Dream Engine can run simultaneously (SQLite WAL mode + igraph auto-resync via `PRAGMA data_version`)
- **Hermes + CerebroCortex coexist** — CC doesn't replace Hermes's built-in MEMORY.md, session search, or skills system. It adds decay, semantic search, associative graphs, and consolidation on top

---

## Links

- **CerebroCortex repo**: [github.com/buckster123/CerebroCortex](https://github.com/buckster123/CerebroCortex)
- **PyPI**: [pypi.org/project/cerebro-cortex](https://pypi.org/project/cerebro-cortex/)
- **Hermes Agent**: [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
- **Memory Provider Plugin PR**: [PR #7913](https://github.com/NousResearch/hermes-agent/pull/7913)
- **Integration Report**: [HERMES_INTEGRATION_REPORT.md](HERMES_INTEGRATION_REPORT.md)
- **Dashboard**: Run `cerebro-api` and visit `http://localhost:8767/ui`
