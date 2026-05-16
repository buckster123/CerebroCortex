# Cerebro Cognitive Bootstrap System (CCBS)

**Status:** Design phase — pivoting from ECS ingestion to bespoke native framework  
**Origin:** Andre's vision for modular session-start priming using Cerebro's native memory system  
**License:** MIT (ours, no restrictions)  
**Version:** 0.1.0-design

---

## 1. Core Concept

A modular cognitive framework stored **inside** CerebroCortex, loaded dynamically at session start based on query analysis + manual triggers. Replaces static system prompts with a **living, gated, recallable cognitive architecture**.

**Key insight:** Cerebro already has everything needed:
- `MemoryType.PROCEDURAL` = cognitive workflows/how-to
- `MemoryType.SEMANTIC` = knowledge/concepts
- `MemoryType.EPISODIC` = session context/history
- GatingEngine = filters what loads
- Episode tracking = maintains session coherence
- Recall scoring = ranks modules by relevance

**Why this beats ECS:**
- ECS is static text injected into context. CCBS is **recalled from memory** based on relevance.
- ECS burns tokens regardless of need. CCBS loads **only what the session requires**.
- ECS is external dependency. CCBS is **self-hosted, self-evolving**.
- ECS has 7 rigid layers. CCBS has **modular, composable cognitive modules**.

---

## 2. Architecture

### 2.1 The SOUL.md — Master Bootstrap Document

A single `MemoryType.SEMANTIC` memory tagged `["soul", "bootstrap", "master"]` that defines:
- The CCBS framework overview
- Available cognitive modules and their purposes
- Loading rules (auto vs manual)
- Default mode configuration
- Manual trigger phrases (e.g. "full load", "debug mode", "creative mode")

**Stored as:** `soul-md` memory in Cerebro  
**Loaded:** Always, at session start (small, ~500 tokens)  
**Function:** The "operating manual" for the bootstrap system

### 2.2 Cognitive Modules

Each module is a `MemoryType.PROCEDURAL` memory with structured tags:

| Module | Purpose | Tags | MemoryType |
|--------|---------|------|-----------|
| `module-core` | Base reasoning patterns, identity, values | `["module", "core", "always"]` | PROCEDURAL |
| `module-analysis` | Analytical thinking, debugging, evaluation | `["module", "analysis"]` | PROCEDURAL |
| `module-creative` | Ideation, lateral thinking, design | `["module", "creative"]` | PROCEDURAL |
| `module-technical` | Code, systems, architecture | `["module", "technical"]` | PROCEDURAL |
| `module-research` | Literature review, synthesis, evaluation | `["module", "research"]` | PROCEDURAL |
| `module-communicate` | Teaching, explaining, documentation | `["module", "communicate"]` | PROCEDURAL |
| `module-meta` | Self-reflection, planning, evaluation | `["module", "meta"]` | PROCEDURAL |
| `module-cerebro` | Cerebro-specific operations, memory management | `["module", "cerebro"]` | PROCEDURAL |
| `knowledge-domain-*` | Domain-specific knowledge bases | `["knowledge", "domain"]` | SEMANTIC |

### 2.3 Loading Strategy

**Layer 1: Mandatory (always loaded)**
- `soul-md` (the bootstrap manifest)
- `module-core` (base identity)

**Layer 2: Auto-detected (query analysis)**
Analyze the user's opening query for intent signals:
- Code/technical keywords → load `module-technical`
- "Design", "create", "ideate" → load `module-creative`
- "Debug", "fix", "error" → load `module-analysis`
- "Research", "paper", "study" → load `module-research`
- "Explain", "teach", "doc" → load `module-communicate`
- "Plan", "evaluate", "reflect" → load `module-meta`
- "Memory", "remember", "recall" → load `module-cerebro`

**Layer 3: Manual overrides**
User can force modules via trigger phrases:
- `"Full load"` or `"Max brain"` → load ALL modules
- `"Debug mode"` → load `module-analysis` + `module-technical`
- `"Creative mode"` → load `module-creative` + suppress `module-analysis`
- `"Cerebro mode"` → load `module-cerebro` + memory management rules
- `"Solo core"` → load only `module-core` (minimal)

**Layer 4: Contextual boost**
If query contains known concepts from Cerebro's knowledge base, boost relevant `knowledge-domain-*` modules.

### 2.4 Token Budget Management

| Mode | Modules | Est. Tokens | Use Case |
|------|---------|-------------|----------|
| Minimal | soul + core only | ~800 | Quick queries, low context |
| Standard | soul + core + 1-2 detected | ~1,500 | Normal sessions |
| Full | soul + core + all modules | ~4,000 | Deep work, complex tasks |
| Custom | soul + user-specified | varies | Specific needs |

**Gating rule:** If context window is tight (<30% remaining), auto-downgrade to Standard.

---

## 3. Memory Storage Format

Each module is stored as a Cerebro memory with structured content:

```markdown
# Module: {name}
## Purpose
One-line description of what this module enables.

## Activation Signals
Keywords/phrases that trigger auto-loading:
- signal1, signal2, signal3

## Cognitive Patterns
The actual instruction content:
1. Pattern name: description
2. Pattern name: description

## Inhibitions (what to AVOID)
- Anti-pattern 1
- Anti-pattern 2

## Related Modules
- module-name (why related)

## Version
0.1.0
```

**Example — module-core:**
```markdown
# Module: core
## Purpose
Base reasoning identity for all sessions.

## Activation Signals
(always loaded)

## Cognitive Patterns
1. Be direct and thorough. Never rush to conclusions.
2. Present options as (a)(b)(c) with alphabetical preference.
3. Prioritize "known-working now" over bleeding-edge.
4. Complete ALL planned phases before declaring done.
5. Use rigorous measurement and numbers in findings.

## Inhibitions
- Never say "that's it" or wrap up early
- Never present single-option recommendations
- Never skip verification steps

## Related Modules
- module-meta (for self-reflection)
- module-technical (for code work)

## Version
0.1.0
```

---

## 4. Session Start Flow

```
User sends query
    ↓
[THALAMUS] Query analysis — extract intent, keywords, domain
    ↓
[RECALL] soul-md (always) + core (always)
    ↓
[SEMANTIC] Match query against module activation signals
    ↓
[GATING] Check token budget, filter modules by priority
    ↓
[ASSEMBLY] Build system prompt block from loaded modules
    ↓
[EPISODE] Start episode tracking with loaded module list
    ↓
Respond to user with module summary (optional)
```

---

## 5. Manual Triggers

Triggers are detected BEFORE query analysis. If found, override auto-detection:

| Trigger | Action | Example |
|---------|--------|---------|
| `"Full load"` / `"Max brain"` / `"All in"` | Load ALL modules | "Full load, let's architect this" |
| `"Solo core"` / `"Minimal"` | Only soul + core | "Solo core — quick check" |
| `"Debug mode"` | core + analysis + technical | "Debug mode: why is this failing?" |
| `"Creative mode"` | core + creative (suppress analysis) | "Creative mode: design a logo" |
| `"Research mode"` | core + research + analysis | "Research mode: what's the state of X?" |
| `"Cerebro mode"` | core + cerebro + meta | "Cerebro mode: optimize my memory" |
| `"Teach me"` / `"Explain"` | core + communicate | "Teach me how X works" |

---

## 6. Integration with Cerebro Features

| Cerebro Feature | CCBS Usage |
|-----------------|------------|
| `GatingEngine` | Filters which modules pass based on query relevance + token budget |
| `SemanticEngine` | Extracts concepts from query to match against module knowledge domains |
| `Episode tracking` | Records which modules were loaded per session for pattern analysis |
| `AffectEngine` | Adjusts tone based on loaded modules (creative = warm, analysis = clinical) |
| `SchemaEngine` | Extracts recurring module combinations as schemas ("debug sessions always load X+Y") |
| `Audit logging` | Logs module loading decisions for transparency |
| `Memory versioning` | Modules are versioned; old versions still accessible |

---

## 7. Implementation Plan

### Phase 1: Bootstrap Infrastructure (Session 1)
1. Create `module-core` and `soul-md` memories in Cerebro
2. Build `scripts/bootstrap_ccbs.py` — one-time setup script
3. Create 3-4 additional modules (analysis, creative, technical)
4. Test loading via `recall()` with tag filtering

### Phase 2: Query Analysis + Auto-Loading (Session 2)
1. Implement intent detection (keyword matching → module selection)
2. Build token budget calculator
3. Test auto-loading with sample queries
4. Add module summary display

### Phase 3: Manual Triggers + Skill (Session 3)
1. Implement trigger phrase detection
2. Build Hermes skill `cerebro-cognitive-bootstrap`
3. Add mode switching (standard/full/custom)
4. Test end-to-end: query → analyze → load → respond

### Phase 4: Evolution (Session 4+)
1. Let dream engine extract schemas from session history
2. Auto-suggest module combinations based on past successful sessions
3. Build "module effectiveness" tracking (which combos produce best outcomes)
4. User-driven module creation UI

---

## 8. Open Design Questions

1. **Module content:** Should we write these from scratch, or derive from ECS inspiration (but reworded, license-clean)?
2. **Query analysis depth:** Simple keyword matching, or semantic similarity search against module content?
3. **Token budget:** Fixed per-mode, or dynamic based on available context window?
4. **Persistence:** Should loaded modules be stored as an episode, or as a session note?
5. **Cross-session learning:** Should CCBS remember which module combos worked best for Andre?
6. **Hermes integration:** Should this be a skill, or built into Cerebro's API server?

---

## 9. Comparison: ECS vs CCBS

| Dimension | ECS (external) | CCBS (native) |
|-----------|---------------|---------------|
| Storage | Static files in repo | Living memories in Cerebro |
| Loading | All-or-nothing | Dynamic, query-aware |
| License | CC BY-NC-SA (restricted) | MIT (ours) |
| Evolution | Manual updates only | Auto-evolves via dream engine |
| Token cost | Fixed ~15K | Variable 0.8K–4K |
| Integration | Injected into prompt | Recalled from memory system |
| Personalization | None | Learns from Andre's patterns |
| Gating | None | Thalamic relevance filtering |
| Multi-session | None | Episode tracking across sessions |

---

## 10. Next Steps

**Andre's decisions needed:**

(a) **Module content source:** Write from scratch, or take ECS inspiration and rephrase (clean room)?

(b) **Integration point:** Hermes skill, Cerebro API endpoint, or both?

(c) **Initial module set:** Start with 4-5 modules, or go bigger?

(d) **Query analysis:** Simple keyword matching for v0.1, or semantic similarity from day one?

(e) **Immediate action:** Shall we start Phase 1 now — create `soul-md` + `module-core` + bootstrap script?
