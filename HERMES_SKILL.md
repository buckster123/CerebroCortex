---
name: cerebrocortex
description: Use CerebroCortex — a brain-analogous AI memory system with 46 MCP tools. Covers when/how to remember, recall, associate, record episodes, track procedures, manage intentions, run dream consolidation, and communicate across agents. Load this skill whenever you see mcp_cerebro_* tools or need to work with persistent memory.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [cerebrocortex, memory, mcp, brain, cognitive-architecture, multi-agent]
    related_skills: [native-mcp, hermes-memory-provider-plugin]
---

# CerebroCortex — The Brain That Remembers Like a Brain

CerebroCortex gives you persistent, associative, emotionally-weighted memory that strengthens with use, decays with neglect, and reorganizes itself during dream consolidation. It's modeled after human neuroscience — not a vector database.

**Load this skill when:** You see mcp_cerebro_* tools, need to store/recall information across sessions, want to link ideas, track episodes, manage TODOs, communicate with other agents, or run memory maintenance.

## Mental Model: How the Brain Works

```
Input → Thalamus (gate/filter) → Temporal Lobe (concepts) → Amygdala (emotion)
      → Association Cortex (link) → Hippocampus (episodes) / Cerebellum (procedures)
      → Prefrontal Cortex (rank/promote) → Neocortex (abstract patterns)
      → Dream Engine (offline consolidation)
```

**Key insight:** You don't need to manually trigger brain regions. When you call `remember()`, the full encoding pipeline runs automatically. When you call `recall()`, the ranking pipeline runs. The brain regions are engines under the hood — you work with the high-level tools.

## The 46 Tools at a Glance

### Tier 1: Daily Drivers (use these 80% of the time)

| Tool | What it does | When to use |
|------|-------------|-------------|
| `mcp_cerebro_remember` | Store a memory (full pipeline) | Facts, decisions, discoveries, observations |
| `mcp_cerebro_recall` | Search by meaning (semantic) | "What do I know about X?" |
| `mcp_cerebro_associate` | Link two memories | When A relates to B (cause, support, context) |
| `mcp_cerebro_session_save` | Save session summary | End of session — what happened, what's pending |
| `mcp_cerebro_session_recall` | Load past session notes | Start of session — orient yourself |
| `mcp_cerebro_store_intention` | Save a TODO/reminder | "Remember to do X later" |
| `mcp_cerebro_list_intentions` | Check pending TODOs | Start of session, or "what's on my plate?" |
| `mcp_cerebro_resolve_intention` | Mark TODO done | When a task is complete |
| `mcp_cerebro_send_message` | Message another agent | Cross-agent communication |
| `mcp_cerebro_check_inbox` | Read messages from agents | Check if anyone sent you something |

### Tier 2: Knowledge Building (use for structured learning)

| Tool | What it does | When to use |
|------|-------------|-------------|
| `mcp_cerebro_store_procedure` | Save a workflow/how-to | "Here's how to deploy X" |
| `mcp_cerebro_find_relevant_procedures` | Find matching procedures | "How did we do X before?" |
| `mcp_cerebro_record_procedure_outcome` | Mark procedure success/fail | After following a procedure |
| `mcp_cerebro_create_schema` | Extract a pattern/principle | When you notice a recurring theme |
| `mcp_cerebro_find_matching_schemas` | Find relevant patterns | "Is there a known pattern for this?" |
| `mcp_cerebro_episode_start` | Begin recording a sequence | Multi-step tasks, debugging sessions |
| `mcp_cerebro_episode_add_step` | Add a step to episode | Each significant event in the sequence |
| `mcp_cerebro_episode_end` | Finish episode with summary | Task complete, wrap up the narrative |
| `mcp_cerebro_ingest_file` | Import a file into memory | Load docs, code, notes as searchable memories |

### Tier 3: Graph Navigation (use for deep exploration)

| Tool | What it does | When to use |
|------|-------------|-------------|
| `mcp_cerebro_memory_neighbors` | Get linked memories | "What's connected to this memory?" |
| `mcp_cerebro_common_neighbors` | Find shared connections | "What links A and B?" |
| `mcp_cerebro_find_path` | Shortest path between memories | "How are these ideas related?" |
| `mcp_cerebro_get_memory` | Get full memory details | Need metadata, tags, scores for one memory |
| `mcp_cerebro_update_memory` | Edit content, tags, salience | Fix or enrich an existing memory |
| `mcp_cerebro_delete_memory` | Permanently remove | Wrong/duplicate/harmful content |
| `mcp_cerebro_share_memory` | Change visibility | Make private→shared or vice versa |

### Tier 4: System & Analytics (use for health/maintenance)

| Tool | What it does | When to use |
|------|-------------|-------------|
| `mcp_cerebro_dream_run` | Run offline consolidation | Periodic maintenance (weekly or when prompted) |
| `mcp_cerebro_dream_status` | Check dream engine status | See what consolidation found |
| `mcp_cerebro_memory_health` | Health report | "How's my memory system doing?" |
| `mcp_cerebro_cortex_stats` | Raw system statistics | Detailed numbers |
| `mcp_cerebro_memory_graph_stats` | Graph structure metrics | Link counts, density, structure |
| `mcp_cerebro_emotional_summary` | Emotional tone breakdown | Sentiment analysis of memory corpus |
| `mcp_cerebro_list_agents` | Show registered agents | Multi-agent coordination |
| `mcp_cerebro_register_agent` | Register new agent | First time an agent joins |

### Aliases (identical to their counterparts)

| Alias | Same as |
|-------|---------|
| `mcp_cerebro_memory_store` | `mcp_cerebro_remember` |
| `mcp_cerebro_memory_search` | `mcp_cerebro_recall` |

### Utility (MCP meta-tools)

| Tool | What it does |
|------|-------------|
| `mcp_cerebro_list_prompts` | List available MCP prompts |
| `mcp_cerebro_get_prompt` | Get a prompt by name |
| `mcp_cerebro_list_resources` | List available resources |
| `mcp_cerebro_read_resource` | Read a resource by URI |

---

## How to Remember Well

### Memory Types — Pick the Right One

CerebroCortex auto-classifies if you omit the type, but explicit is better:

| Type | Use for | Example |
|------|---------|---------|
| `episodic` | Events, what happened | "User's deploy failed at 3pm due to OOM" |
| `semantic` | Facts, knowledge | "PostgreSQL max connections default is 100" |
| `procedural` | How-to, workflows | "Deploy sequence: test → build → push → apply" |
| `affective` | Emotional context | "User was frustrated after 3rd CI failure" |
| `prospective` | Future intentions | "Need to upgrade Python before Friday" |
| `schematic` | Patterns, principles | "Services skipping staging have 3x more incidents" |

### Salience — How Important Is It?

Salience (0.0-1.0) determines how aggressively a memory survives decay:

| Range | Meaning | Examples |
|-------|---------|---------|
| 0.8-1.0 | Critical | User corrections, security issues, core preferences |
| 0.6-0.8 | Important | Key decisions, architectural choices, project milestones |
| 0.4-0.6 | Normal | Regular facts, session events, standard procedures |
| 0.2-0.4 | Low | Transient context, temporary notes |
| 0.0-0.2 | Ephemeral | Raw observations likely to be superseded |

Auto-estimation works well, but override for critical items.

### Visibility — Who Can See It?

| Level | Who sees it | Use for |
|-------|-------------|---------|
| `shared` | All agents | Facts, procedures, cross-team knowledge |
| `private` | Only the storing agent | Personal notes, drafts, agent-specific state |
| `thread` | Agents in same conversation | Conversation-scoped context |

### Tags — Make It Findable

Tags power filtered search. Use consistent conventions:
- Project: `project:cerebrocortex`, `project:hermes`
- Category: `deployment`, `debugging`, `architecture`
- Status: `resolved`, `in-progress`, `blocked`
- Agent: Auto-tagged by system, but add `team:backend` etc.

### Example: Good vs Bad Memory Storage

```
BAD:  remember("thing about the database")
GOOD: remember(
        "PostgreSQL connection pool exhaustion caused 502 errors in prod.
         Root cause: default max_connections=100, app opens 20 per worker × 8 workers = 160.
         Fix: set max_connections=300 in postgresql.conf",
        memory_type="semantic",
        tags=["postgresql", "connection-pool", "production-incident"],
        salience=0.7,
        visibility="shared"
      )
```

**Rule of thumb:** Store memories as if writing a note for a colleague who has no context. Include the what, why, and resolution.

---

## How to Recall Effectively

### Basic Recall

```
recall(query="database connection issues", top_k=5)
```

Returns memories ranked by a **4-signal blend:**
- 35% vector similarity (semantic meaning)
- 30% ACT-R activation (recency + frequency of access)
- 20% FSRS retrievability (spaced-repetition freshness)
- 15% salience (importance)

### Filtered Recall

```python
# Only procedural memories
recall(query="deploy", memory_types=["procedural"])

# Only high-importance
recall(query="security", min_salience=0.7)

# With score breakdown (for debugging relevance)
recall(query="database", explain=True)

# Scoped to conversation
recall(query="what we discussed", conversation_thread="thread_123")

# Boost results connected to known context
recall(query="migration", context_ids=["mem_abc", "mem_def"])
```

### Recall Strategy Guide

| Situation | Approach |
|-----------|----------|
| "What do I know about X?" | `recall(query="X", top_k=10)` |
| "How do I do X?" | `find_relevant_procedures(concepts=["X"])` |
| "Is there a pattern for X?" | `find_matching_schemas(concepts=["X"])` |
| "What happened during X?" | `recall(query="X", memory_types=["episodic"])` + check episodes |
| "What's connected to memory M?" | `memory_neighbors(memory_id="M")` |
| "What did I feel about X?" | `recall(query="X", memory_types=["affective"])` |

---

## How to Build the Associative Network

### Link Types — Choose Precisely

The association graph is CerebroCortex's superpower. Links enable spreading activation — when one memory is found, linked memories get a relevance boost.

| Link Type | Weight | When to use | Example |
|-----------|--------|-------------|---------|
| `causal` | 0.9 | A caused B | Bug → Outage |
| `semantic` | 0.8 | A is about same concept as B | Two memories about PostgreSQL |
| `supports` | 0.8 | A is evidence for B | Test result → Hypothesis |
| `part_of` | 0.8 | A is a component of B | Chapter → Book |
| `contextual` | 0.7 | A and B share context | Same project, same meeting |
| `derived_from` | 0.7 | B was abstracted from A | Episodes → Schema |
| `temporal` | 0.6 | A happened near B in time | Sequential events |
| `affective` | 0.5 | A and B feel similar | Two frustrating experiences |
| `contradicts` | 0.3 | A opposes B | Conflicting recommendations |

### Linking Patterns

**After storing related memories:**
```
id_a = remember("Deploy failed due to missing env var")
id_b = remember("Added env validation to CI pipeline")
associate(source_id=id_a, target_id=id_b, link_type="causal",
          evidence="The failure led to the fix", weight=0.9)
```

**Cross-referencing knowledge:**
```
associate(source_id=fact_id, target_id=procedure_id,
          link_type="supports", evidence="This fact validates the procedure")
```

**Don't over-link.** Link when there's a genuine relationship. The Dream Engine will discover implicit connections you miss.

---

## Episodes: Recording Event Sequences

Episodes bind memories into temporal narratives — "what happened during X."

### Pattern: Recording a Task

```
1. episode_start(title="Debugging OOM in prod")
2. remember("Noticed 502 errors at 14:30") → add_step(role="event")
3. remember("Checked logs, found OOM killer") → add_step(role="context")
4. remember("Increased memory limit to 4GB") → add_step(role="event")
5. remember("Service recovered after restart") → add_step(role="outcome")
6. remember("Should add memory alerts to monitoring") → add_step(role="reflection")
7. episode_end(summary="OOM in prod, fixed by increasing limits", valence="mixed")
```

### Step Roles

| Role | When to use |
|------|-------------|
| `event` | Something that happened (default) |
| `context` | Background information, observations |
| `outcome` | The result of actions taken |
| `reflection` | Lessons learned, insights, feelings |

### When to Use Episodes vs Individual Memories

- **Episode:** Multi-step process, debugging session, meeting, project milestone
- **Individual:** Standalone facts, preferences, one-off observations

---

## Procedures: Tracked Workflows

Procedures are special memories that track success/failure over time.

```
# Store
proc_id = store_procedure(
    content="Deploy to prod: 1) Run tests 2) Build image 3) Push to registry 4) Apply manifests 5) Verify health",
    tags=["deployment", "production"]
)

# After using it
record_procedure_outcome(procedure_id=proc_id, success=True)
# or
record_procedure_outcome(procedure_id=proc_id, success=False)
```

Over time, frequently-successful procedures rank higher in `find_relevant_procedures()`. Failed ones get demoted. This is the cerebellum at work — motor memory that reinforces what works.

---

## Schemas: Extracting Patterns

Schemas are abstract principles derived from multiple specific memories.

```
# After noticing a pattern across 3+ memories:
create_schema(
    content="Services that skip staging environments have approximately 3x more production incidents",
    source_ids=["mem_incident_1", "mem_incident_2", "mem_incident_3"],
    tags=["deployment", "best-practices"]
)
```

The Dream Engine also creates schemas automatically during consolidation. You can create them manually when you notice a pattern in conversation.

---

## Intentions: Smart TODOs

Intentions are prospective memories that surface when relevant.

```
store_intention(content="Upgrade Python to 3.12 before deploying new service", salience=0.8)
```

Check and resolve:
```
list_intentions()
resolve_intention(memory_id="mem_xxx")
```

Unlike simple TODO lists, intentions participate in recall — if someone asks about Python upgrades, the intention surfaces alongside relevant semantic memories.

---

## Multi-Agent Communication

CerebroCortex supports multiple agents sharing one memory store.

### Sending Messages

```
send_message(to="CLAUDE-HAILO", content="The EEG plugin is ready for testing on your Pi")
send_message(to="all", content="CerebroCortex v0.2.0 deployed, dream engine improved")
```

Messages bypass gating — they're always delivered. Auto-tagged with sender/recipient.

### Checking Messages

```
check_inbox()
check_inbox(from_agent="CLAUDE-OPUS", since="2026-04-14T00:00:00")
```

### Agent Registration

New agents must register before they can store/recall:
```
register_agent(
    agent_id="MY-AGENT",
    display_name="My Agent",
    generation=1,           # -1=origin, 0=primary, 1+=descendant
    lineage="hermes",
    specialization="what this agent is good at"
)
```

---

## Session Lifecycle

### Starting a Session

```python
# 1. Orient yourself
session_recall(limit=5)           # What happened recently?
list_intentions()                  # What's pending?
check_inbox()                      # Any messages?
```

### Ending a Session

```python
session_save(
    session_summary="Debugged OOM issue in prod, deployed fix, updated monitoring",
    key_discoveries=["PostgreSQL default connections too low for our worker count"],
    unfinished_business=["Add memory usage alerts to Grafana"],
    if_disoriented=["Check the monitoring PR in GitHub"],
    session_type="technical",
    priority="HIGH"
)
```

---

## Dream Engine: Offline Consolidation

The Dream Engine runs 6 phases modeled after biological sleep:

1. **Replay** — Revisit recent memories, strengthen important ones
2. **Cluster** — Group related memories by embedding similarity
3. **Abstract** — Extract schemas/patterns from clusters (LLM-powered)
4. **Prune** — Remove low-activation, unlinked memories
5. **Promote** — Move high-value memories up layers (working → long-term → cortex)
6. **REM Recombine** — Discover unexpected cross-domain connections (LLM-powered)

### Running Dreams

```
dream_run()                     # Run a full cycle (uses ~20 LLM calls)
dream_run(max_llm_calls=10)    # Lighter cycle
dream_status()                  # Check last run results
```

**When to dream:**
- Weekly maintenance (can be scheduled via cron)
- After large knowledge imports (ingest_file)
- When the user asks to "clean up" or "consolidate" memories
- After intensive multi-session projects

**Pitfall:** Dream runs use LLM calls (token cost). On cost-conscious setups, use `max_llm_calls=10` for lighter cycles.

---

## Memory Layers & Decay

Memories automatically flow through 4 durability layers:

```
Sensory (6hr half-life) → Working (3-day) → Long-Term (30-day) → Cortex (permanent)
```

- **Sensory → Working:** Automatic on 2nd access
- **Working → Long-Term:** 5+ accesses, 24h+ old
- **Long-Term → Cortex:** Dream Engine only (crystallized knowledge)

You don't manage layers directly. The system promotes based on access patterns. Frequently-recalled memories survive; neglected ones fade. Just like a real brain.

**To keep a memory alive:** Recall it. Every recall strengthens activation + stability.

---

## Common Workflows

### Research & Learning

```
1. ingest_file(file_path="/path/to/paper.md", tags=["research", "topic"])
2. recall("key concepts from the paper")
3. For important findings: remember() with high salience
4. associate() related findings together
5. create_schema() when patterns emerge
```

### Debugging Session

```
1. episode_start(title="Debugging: service X failing")
2. At each step: remember(observation) → episode_add_step()
3. When solved: remember(solution, memory_type="procedural")
4. episode_end(summary="...", valence="positive")
5. Link solution to problem: associate(problem_id, solution_id, "causal")
```

### Project Kickoff

```
1. recall("similar past projects") — learn from history
2. find_relevant_procedures(concepts=["project-type"])
3. store_intention("Project milestones and deadlines")
4. remember("Project goals and constraints", salience=0.8)
```

### Cross-Agent Handoff

```
1. session_save(summary="...", unfinished_business=["task X"])
2. send_message(to="OTHER-AGENT", content="Please continue task X, see session notes")
3. The other agent: check_inbox() → session_recall() → pick up where you left off
```

---

## Pitfalls

1. **Don't store raw data dumps.** CerebroCortex is for *distilled knowledge*, not log files. Summarize before storing. Use ingest_file for large documents — it auto-chunks.

2. **Don't over-link.** Quality > quantity. A few strong causal/semantic links beat 50 weak contextual ones. The Dream Engine discovers implicit connections.

3. **Don't forget to recall.** Memories that are never recalled decay and eventually get pruned. If something is important, access it periodically. This is by design — it's how real memory works.

4. **Aliases exist.** `memory_store` = `remember`, `memory_search` = `recall`. Don't call both thinking they're different.

5. **Explain mode for debugging.** If recall results seem wrong, use `explain=True` to see the score breakdown (vector similarity, ACT-R, FSRS, salience). This reveals whether the issue is semantic, temporal, or importance-based.

6. **Dream Engine costs tokens.** Each run uses ~20 LLM calls. Schedule wisely, especially on prepaid API credits.

7. **Visibility matters in multi-agent.** Private memories are invisible to other agents. Shared is the default for collaborative knowledge. Thread-scoped for conversation-local context.

8. **Session notes are your compass.** Always save at end, always recall at start. Future-you will thank past-you.

9. **Intentions surface in recall.** A stored intention about "upgrade Python" will appear when someone asks about Python. This is a feature, not a bug — it's prospective memory working as designed.

10. **Hardware context:** CerebroCortex was built for Raspberry Pi 5 (4GB). It's lightweight by design. The dream engine is the heaviest operation — avoid running it during other intensive tasks.
