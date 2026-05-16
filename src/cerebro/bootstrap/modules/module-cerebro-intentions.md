# Module: cerebro-intentions
## Purpose
Prospective memory management — TODOs, reminders, and deferred tasks that participate in recall.

## Activation Signals
- "todo", "plan", "remember to", "don't forget", "later", "next", "after this"
- Any deferred task or future action

## Cognitive Patterns
1. **store_intention()** — Save a TODO or reminder.
   - Unlike simple lists, intentions participate in recall — they surface when relevant.
   - Set salience high (0.8+) for urgent items.
   - Tag with domain: ["bug", "feature", "research", "infrastructure"].

2. **list_intentions()** — Check pending TODOs.
   - Call at start of session to see what's on the plate.
   - Filter by `min_salience` to focus on important items.

3. **resolve_intention()** — Mark TODO done.
   - Call immediately when a task completes.
   - Unresolved intentions clutter recall — keep the queue clean.

4. **Integration with episodes** — Link intentions to episodes.
   - When starting work on a TODO, create an episode for it.
   - When the episode ends, resolve the intention.

## Inhibitions
- Never store TODOs in harness memory — Cerebro intentions are searchable and recallable.
- Never leave intentions unresolved for weeks — prune stale ones.
- Never create intentions for trivial one-off tasks — use them for meaningful deferred work.

## Related Modules
- module-cerebro-session (episode tracking for intention work)
- module-cerebro-ops (general memory operations)

## Version
0.1.0
