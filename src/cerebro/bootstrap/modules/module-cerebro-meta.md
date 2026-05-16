# Module: cerebro-meta
## Purpose
Meta-cognitive guidance for memory usage. The 90/10 rule, auto-remember triggers, and persistence strategy.

## Activation Signals
(always loaded)

## Cognitive Patterns
1. **The 90/10 Rule** — Cerebro is the primary memory. Harness memory is a small static cache.
   - 90% of memory ops should go to Cerebro (remember/recall/associate).
   - 10% goes to harness memory (host OS, static user prefs, conventions that never change).
   - When in doubt: Cerebro.

2. **What goes to harness memory** (static, <200 chars, never changes):
   - Host OS and hardware facts.
   - User communication style preferences.
   - Stable conventions (e.g., "always use venv", "pytest with xdist").
   - What does NOT go there: versions, project states, task progress, session outcomes.

3. **What goes to Cerebro** (growing, changing, recallable):
   - Session summaries and key discoveries.
   - Project architecture decisions.
   - Bug fixes and their solutions.
   - TODOs and intentions.
   - Cross-project relationships and integration notes.
   - Any fact that might be needed in a future session.

4. **Auto-remember triggers** — Automatically persist without waiting for user prompt:
   - After any successful end-to-end verification.
   - At every phase boundary on medium+ complexity tasks.
   - When user says "remember this" or "don't do that again."
   - When discovering a new API quirk, workflow, or convention.

5. **Memory ops preference order**:
   - (a) `mcp_cerebro_remember` — for facts and decisions.
   - (b) `mcp_cerebro_session_save` — for session handovers.
   - (c) `mcp_cerebro_store_intention` — for TODOs.
   - (d) `memory` tool — ONLY for static host/user facts <200 chars.

6. **Density economics** — Andre prefers sub-1K handovers over bigger context windows.
   - Keep memories information-dense.
   - Use bullet tables over prose when listing facts.
   - Latent-space notation and symbolic compression are OK.

## Inhibitions
- Never grow harness memory files — they have a ~2.2K char limit and cause truncation.
- Never store versions, commit SHAs, or PR numbers in harness memory — stale in 7 days.
- Never wait to be asked to save state — proactive persistence is the default.

## Related Modules
- module-cerebro-ops (how to execute memory ops)
- module-cerebro-session (when to checkpoint)
- module-core (base identity)

## Version
0.1.0
