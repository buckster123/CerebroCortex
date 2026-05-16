# Module: cerebro-agents
## Purpose
Multi-agent awareness and cross-agent communication via CerebroCortex.

## Activation Signals
- "agent", "message", "hailo", "apex", "send to", "check inbox"
- Any cross-instance or multi-agent context

## Cognitive Patterns
1. **Registered agents** — Cerebro knows three agents sharing one store:
   - CLAUDE-OPUS (this host, Krackan)
   - CLAUDE-HAILO (Hailo Pi, edge inference)
   - CLAUDE-APEX (ApexAurum cloud instance)

2. **send_message()** — Send a message to another agent.
   - Use `to="all"` for broadcast.
   - Messages are stored as SEMANTIC memories with `from:{agent}` / `to:{agent}` tags.
   - Always deliver important findings to relevant agents.

3. **check_inbox()** — Check for messages from other agents.
   - Call at session start to see if other agents left updates.
   - Filter by `from_agent` to see messages from a specific agent.

4. **Cross-agent conventions** —
   - Shared memories use visibility="shared".
   - Agent-specific memories use visibility="private" with agent_id set.
   - Messages auto-tag with from/to and link replies.

5. **MCP-to-REST shim** — For remote agents without local Cerebro code.
   - REST API at `http://localhost:8767` exposes full functionality.
   - Remote agents can use HTTP instead of MCP stdio.

## Inhibitions
- Never assume another agent has the same context — always include relevant background in messages.
- Never send raw data dumps to other agents — summarize and link to Cerebro memories.
- Never ignore inbox messages — they may contain critical updates.

## Related Modules
- module-cerebro-ops (for storing shared memories)
- module-cerebro-index (for finding the right module)

## Version
0.1.0
