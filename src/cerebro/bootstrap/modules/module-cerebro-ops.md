# Module: cerebro-ops
## Purpose
Core memory operations. How to store, retrieve, link, and search memories in CerebroCortex.

## Activation Signals
(always loaded)

## Cognitive Patterns
1. **remember()** — Store facts, decisions, discoveries, and anything worth recalling later.
   - Use tags for categorization: ["project:X", "bug", "decision", "architecture"]
   - Set salience 0.7+ for important facts, 0.9+ for critical ones.
   - Set visibility: "shared" for cross-agent knowledge, "private" for agent-specific.
   - Memory type: "semantic" for facts, "procedural" for workflows, "episodic" for session events.

2. **recall()** — Search memories by semantic similarity.
   - Use `explain=True` when debugging why a memory didn't surface.
   - Filter by `memory_types` to narrow scope.
   - Filter by `agent_id` for agent-specific memories.
   - Use `min_salience` to cut noise.

3. **associate()** — Link related memories.
   - Link types: temporal (same session), causal (caused), semantic (related), supports (evidence), contradicts (conflict), part_of (component), derived_from (origin).
   - Always link new memories to related existing ones.

4. **search_vision()** — Search image memories by text description.
   - Use for finding visually similar memories or image-to-image search.

5. **describe_image()** — Generate text description of an image file.
   - Use before ingesting images to get searchable captions.

6. **Ingestion pipeline** — For files: use `ingest_file()` or the `/ingest/upload` endpoint.
   - Supported: txt, md, json, pdf, html, csv, png, jpg, webp, py, js, ts, go, rs, java, rb, sh.
   - Images get vision embeddings + optional OCR/caption.

## 90/10 Rule in Practice
- BEFORE storing anything in harness memory, ask: "Will this change or grow?"
- If yes → Cerebro remember().
- If no and <200 chars → harness memory is OK.
- If no and >200 chars → Cerebro remember() anyway (density economics).

## Inhibitions
- Never store raw data dumps, file counts, or commit SHAs in Cerebro — these go stale.
- Never skip linking related memories — orphan nodes decay faster.
- Never use harness memory for session summaries, task progress, or TODOs.

## Related Modules
- module-cerebro-session (when tracking sessions)
- module-cerebro-intentions (for TODO storage)
- module-cerebro-meta (for persistence strategy)

## Version
0.1.0
