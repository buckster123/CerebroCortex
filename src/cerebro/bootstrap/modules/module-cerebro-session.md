# Module: cerebro-session
## Purpose
Session tracking and checkpoint protocol. How to maintain continuity across Hermes context windows using CerebroCortex as persistent state.

## Activation Signals
(always loaded)

## Cognitive Patterns
1. **Episode tracking** — Record sequences of related events as episodes.
   - `episode_start(title, tags)` at the beginning of a focused task.
   - `episode_add_step(episode_id, memory_id, role)` to link memories to the episode.
   - `episode_end(episode_id, summary, valence)` when the task completes.
   - Roles: "event" (what happened), "context" (background), "outcome" (result), "reflection" (lessons).

2. **Session checkpoint protocol** — Prevent context loss from Hermes compaction.
   | Complexity | Trigger | Operation | Frequency |
   |------------|---------|-----------|-----------|
   | Low (<5 calls, single file) | End of task | `remember` key fact | Every session |
   | Medium (5-15 calls, multi-file, new skill) | Phase boundary | `session_save` + `remember` | Every 30 min |
   | High (15+ calls, architecture, debugging) | Every milestone | `episode_start/end` + `session_save` | Every 15 min |
   | Critical (cross-project, new integration, VR) | After every verification | `session_save` (HIGH) + `remember` + skill | Immediately |

3. **session_save()** — Durable session summary for future self.
   - Include: key_discoveries, unfinished_business, if_disoriented notes.
   - Priority: HIGH for architecture/debugging, MEDIUM for normal work.
   - session_type: "technical" for code, "task" for execution, "orientation" for planning.

4. **session_recall()** — Retrieve previous session notes at start.
   - Call with `lookback_hours=168` (1 week) to see recent work.
   - Use `priority_filter="HIGH"` to find critical unfinished tasks.

5. **Before ANY context switch** — Save current state to Cerebro first.
6. **After ANY successful end-to-end verification** — Save immediately (tests pass, API responds, APK installs).

## Inhibitions
- Never let a long session end without at least one `session_save` or `remember`.
- Never rely on harness memory for session continuity — it has ~2.2K char limit.
- Never skip `episode_end()` — stale episodes auto-close after 24h but explicit is better.

## Related Modules
- module-cerebro-ops (for storing memories within episodes)
- module-cerebro-meta (for deciding what to persist)

## Version
0.1.0
