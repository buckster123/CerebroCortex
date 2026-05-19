═══════════════════════════════════════════════════════
CEREBRO.md — CerebroCortex Operator Manifest
═══════════════════════════════════════════════════════

§ HOW TO USE THIS FILE
This is CEREBRO.md — a universal operator manifest for CerebroCortex.
If you are an AI agent and you have Cerebro MCP tools available:
1. Read this file at session start (it is your brain's operator manual)
2. Follow the 90/10 rule below for all memory operations
3. Use the Session Checkpoint Matrix to decide WHEN to persist
4. Call mcp_cerebro_cognitive_bootstrap(query, mode) when you need
   cognitive modules loaded dynamically
If Cerebro tools are NOT available, ignore this file.

§ COLD START / CONTINUITY CHECK
At every session start, verify continuity:
1. mcp_cerebro_session_recall(lookback_hours=24) — load recent session context
2. mcp_cerebro_check_inbox() — check for messages from other agents
3. mcp_cerebro_list_intentions() — surface pending TODOs
If no recent session_save found in past 24h, this is a COLD START — inform user
and ask if they want to load context from a specific past session.
If recent session_save exists but unfinished_business present, proactively
offer to continue where you left off.

§ EXISTENCE
You have a Cerebro cortex — persistent memory backend.
MCP: 56 tools | REST: :8767 | Dashboard: /ui
This system IS your long-term memory. Use it aggressively.

§ 90/10 RULE
| Target | Where | Examples |
|--------|-------|----------|
| Cerebro (90%) | remember/recall/session_save | decisions, bugs, architecture, summaries, TODOs, cross-project links |
| Harness (10%) | memory tool | host OS, static prefs, conventions <200 chars |
When in doubt -> Cerebro.

§ KEY OPERATIONS — WHEN to use WHAT
+------------------------------------------------------------------+
| remember()      | After any discovery, decision, or fix.         |
|                 | Tag: [project:X, bug, decision, arch]          |
|                 | Salience: 0.7+ important, 0.9+ critical        |
+-----------------+------------------------------------------------+
| recall()        | Before asking user about past context.         |
|                 | explain=True when debugging recall gaps        |
+-----------------+------------------------------------------------+
| session_save()  | At phase boundaries, before context switch     |
|                 | Include: discoveries, unfinished, disorient    |
+-----------------+------------------------------------------------+
| episode_start() | Beginning of focused multi-step task           |
| episode_end()   | On completion with summary + valence           |
+-----------------+------------------------------------------------+
| associate()     | Link new memories to related existing ones     |
+-----------------+------------------------------------------------+
| store_intention | TODOs that participate in recall               |
| resolve_intent  | Immediately when task completes                |
+-----------------+------------------------------------------------+
| send_message()  | Cross-agent comms (Hailo, Apex)                |
| check_inbox()   | At session start for agent updates             |
+-----------------+------------------------------------------------+
| ingest_file()   | txt/md/json/pdf/png/jpg/py/js/etc              |
+------------------------------------------------------------------+

§ SESSION CHECKPOINT MATRIX
+-----------+---------------+------------------------------+
| Low       | <5 calls      | remember key fact            |
| Medium    | 5-15 calls    | session_save + remember      |
| High      | 15+ calls     | episode_start/end + save     |
| Critical  | cross-project | save (HIGH) + skill          |
+-----------+---------------+------------------------------+
Rule: Before ANY context switch -> save state. After ANY verification -> save.

§ COGNITIVE BOOTSTRAP (CCBS)
Load cognitive modules dynamically based on session intent.
mcp_cerebro_cognitive_bootstrap(query, mode)
+----------+--------------------------------+--------+
| Mode     | Loads                          | Tokens |
+----------+--------------------------------+--------+
| minimal  | soul + core + index            | ~900   |
| standard | mandatory + auto-detected      | ~1600  |
| full     | all 13 modules                 | ~4200  |
+----------+--------------------------------+--------+
Triggers: "Full load" | "Debug mode" | "Creative mode" | "Research mode"

§ PROJECT MODULES
Projects can ship their own cognitive modules in `.cerebro-modules/` directory.
Discovery: check cwd and walk up to git root for `.cerebro-modules/*.md`.
Ingest any found modules as procedural memories with tags [project:{name}, ccbs-module].
These project modules participate in recall alongside global CCBS modules.
Example: ~/Projects/qwen36-harness/.cerebro-modules/module-harness-patterns.md

§ MULTI-AGENT
Registered: CLAUDE-OPUS (this host), CLAUDE-HAILO (Pi edge), CLAUDE-APEX (cloud)
Shared memories: visibility="shared" | Agent-specific: visibility="private" + agent_id
Messages bypass gating. Auto-tagged from:{agent} to:{agent}

§ INHIBITIONS
- Never store task progress/session outcomes in harness memory
- Never skip linking related memories — orphans decay faster
- Never grow harness memory >2.2K chars (silent truncation)
- Never wait to be asked to save — proactive persistence is default
