"""Dream Engine - Default Mode Network for CerebroCortex.

Runs between sessions (or on-demand) to consolidate memories through
6 biologically-inspired phases:

1. SWS Replay      - Slow-wave sleep: replay episodes, strengthen temporal links
2. Pattern Extract  - Cluster similar memories, extract reusable procedures
3. Schema Formation - Abstract episodes into general principles
4. Emotional Reproc - Adjust salience based on outcomes (negative = higher)
5. Pruning          - Delete noise, decay stale memories, promote strong ones
6. REM Recombine    - Sample diverse memories, discover unexpected connections

LLM-assisted phases (2, 3, 6) use the LLMClient with auto-failover.
Algorithmic phases (1, 4, 5) run without LLM.

Usage:
    dream = DreamEngine(cortex, llm_client)
    report = dream.run_cycle()
"""

import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from cerebro.config import (
    DREAM_CLUSTER_MIN_SIZE,
    DREAM_CLUSTER_SIMILARITY_THRESHOLD,
    DREAM_LLM_BUDGET_PATTERN,
    DREAM_LLM_BUDGET_REM,
    DREAM_LLM_BUDGET_SCHEMA,
    DREAM_MAX_LLM_CALLS,
    DREAM_PRUNING_MAX_SALIENCE,
    DREAM_PRUNING_MIN_AGE_HOURS,
    DREAM_REM_MIN_CONNECTION_STRENGTH,
    DREAM_REM_PAIR_CHECKS,
    DREAM_REM_SAMPLE_SIZE,
)
from cerebro.types import DreamPhase, EmotionalValence, LinkType, MemoryType

logger = logging.getLogger("cerebro-dream")


# =============================================================================
# Dream report data
# =============================================================================

@dataclass
class PhaseReport:
    """Report from a single dream phase."""
    phase: DreamPhase
    memories_processed: int = 0
    links_created: int = 0
    links_strengthened: int = 0
    memories_pruned: int = 0
    schemas_extracted: int = 0
    procedures_extracted: int = 0
    llm_calls: int = 0
    duration_seconds: float = 0.0
    notes: str = ""
    success: bool = True


@dataclass
class DreamReport:
    """Report from a full dream cycle."""
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    agent_id: Optional[str] = None
    phases: list[PhaseReport] = field(default_factory=list)
    episodes_consolidated: int = 0
    total_llm_calls: int = 0
    total_duration_seconds: float = 0.0
    success: bool = True

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "agent_id": self.agent_id,
            "episodes_consolidated": self.episodes_consolidated,
            "total_llm_calls": self.total_llm_calls,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "success": self.success,
            "phases": [
                {
                    "phase": p.phase.value,
                    "memories_processed": p.memories_processed,
                    "links_created": p.links_created,
                    "links_strengthened": p.links_strengthened,
                    "memories_pruned": p.memories_pruned,
                    "schemas_extracted": p.schemas_extracted,
                    "procedures_extracted": p.procedures_extracted,
                    "llm_calls": p.llm_calls,
                    "duration_seconds": round(p.duration_seconds, 2),
                    "notes": p.notes,
                    "success": p.success,
                }
                for p in self.phases
            ],
        }


# =============================================================================
# LLM prompts
# =============================================================================

SYSTEM_DREAM = (
    "You are the Dream Engine of CerebroCortex, a brain-analogous AI memory system. "
    "You process memories during consolidation, extracting patterns, creating schemas, "
    "and finding unexpected connections. Respond in structured JSON only."
)

PROMPT_EXTRACT_PATTERNS = """Analyze these memories and extract reusable patterns or procedures.

Memories:
{memories}

Return a JSON array of extracted patterns. Each pattern should have:
- "content": A clear, actionable procedure or pattern (1-3 sentences)
- "source_indices": Which memory indices (0-based) this pattern comes from
- "tags": Relevant tags for the pattern

Return ONLY valid JSON array. Example:
[{{"content": "When debugging async code, check the event loop first, then verify awaits", "source_indices": [0, 2], "tags": ["debugging", "async"]}}]"""

PROMPT_FORM_SCHEMA = """Analyze these related memories and form an abstract schema (general principle).

Memories:
{memories}

What general principle, pattern, or lesson connects these memories?

Return JSON with:
- "content": The abstract principle (1-2 sentences, general enough to apply beyond these specific cases)
- "tags": Relevant categorization tags

Return ONLY valid JSON object. Example:
{{"content": "Iterative refinement with user feedback produces better results than upfront design", "tags": ["methodology", "development"]}}"""

PROMPT_REM_CONNECT = """You are looking at two seemingly unrelated memories. Find an unexpected but meaningful connection.

Memory A: {memory_a}
Memory B: {memory_b}

Is there a meaningful connection between these? If yes, describe it.

Return JSON with:
- "connected": true/false
- "link_type": One of: semantic, causal, supports, contradicts
- "reason": Brief explanation of the connection (1 sentence)
- "weight": Connection strength 0.0-1.0

Return ONLY valid JSON object."""


# =============================================================================
# Dream Engine
# =============================================================================

class DreamEngine:
    """The Default Mode Network. Consolidates memories through 6 dream phases."""

    def __init__(self, cortex, llm_client=None,
                 on_phase_complete: Optional[Callable[[PhaseReport, str | None], None]] = None):
        """Initialize Dream Engine.

        Args:
            cortex: CerebroCortex instance (must be initialized)
            llm_client: LLMClient instance (optional, LLM phases skipped if None)
            on_phase_complete: Callback(phase_report, agent_id) called after each phase
        """
        self._cortex = cortex
        self._llm = llm_client
        self._on_phase_complete = on_phase_complete
        self._llm_calls_remaining = DREAM_MAX_LLM_CALLS
        self._running = False
        self._agent_id: Optional[str] = None
        self._cycle_id: Optional[str] = None
        self._last_report: Optional[DreamReport] = None
        self._last_reports: list[DreamReport] = []

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_report(self) -> Optional[DreamReport]:
        return self._last_report

    @property
    def last_reports(self) -> list[DreamReport]:
        """All reports from the most recent run_all_agents_cycle() call."""
        return self._last_reports

    # =========================================================================
    # Main cycle
    # =========================================================================

    def run_cycle(
        self,
        agent_id: Optional[str] = None,
        cycle_id: Optional[str] = None,
    ) -> DreamReport:
        """Run a full dream consolidation cycle (all 6 phases).

        Args:
            agent_id: If provided, scope all queries to this agent's visible
                      memories (SHARED + own PRIVATE/THREAD). If None, legacy
                      unscoped behavior.
            cycle_id: Resume a previous incomplete cycle. If None, starts fresh.

        Returns:
            DreamReport with details of each phase.
        """
        if self._running:
            raise RuntimeError("Dream cycle already in progress")

        self._running = True
        self._agent_id = agent_id
        self._llm_calls_remaining = DREAM_MAX_LLM_CALLS
        self._cycle_id = cycle_id or f"dream_{uuid.uuid4().hex[:12]}"
        report = DreamReport(agent_id=agent_id)
        cycle_start = time.time()

        # Check which phases are already done (for resume)
        completed_phases = self._cortex.graph.get_completed_phases(self._cycle_id)
        resuming = bool(completed_phases)

        try:
            scope_label = f" for agent {agent_id}" if agent_id else ""
            resume_label = f" (resuming {len(completed_phases)} done)" if resuming else ""
            logger.info(f"Dream cycle {self._cycle_id} starting{scope_label}{resume_label}...")

            # Pre-phase: auto-close stale episodes
            closed = self._cortex.episodes.close_stale_episodes()
            if closed:
                logger.info(f"Dream pre-phase: auto-closed {len(closed)} stale episodes")

            # Ordered phase dispatch table
            phases = [
                (DreamPhase.SWS_REPLAY, self._phase_sws_replay),
                (DreamPhase.PATTERN_EXTRACTION, self._phase_pattern_extraction),
                (DreamPhase.SCHEMA_FORMATION, self._phase_schema_formation),
                (DreamPhase.EMOTIONAL_REPROCESSING, self._phase_emotional_reprocessing),
                (DreamPhase.PRUNING, self._phase_pruning),
                (DreamPhase.REM_RECOMBINATION, self._phase_rem_recombination),
            ]

            for phase_enum, phase_fn in phases:
                if phase_enum.value in completed_phases:
                    logger.info(f"Skipping {phase_enum.value} (already completed in {self._cycle_id})")
                    continue
                phase_report = phase_fn()
                report.phases.append(phase_report)
                self._notify_phase(phase_report)

            # Mark episodes consolidated
            episodes = self._cortex.episodes.get_unconsolidated(agent_id=self._agent_id)
            for ep in episodes:
                self._cortex.episodes.mark_consolidated(ep.id)
            report.episodes_consolidated = len(episodes)

            report.success = all(p.success for p in report.phases)
            logger.info(
                f"Dream cycle {self._cycle_id} complete{scope_label}: "
                f"{len(report.phases)} phases run, "
                f"{report.episodes_consolidated} episodes consolidated"
            )

        except Exception as e:
            logger.error(f"Dream cycle {self._cycle_id} failed: {e}", exc_info=True)
            report.success = False

        finally:
            self._running = False
            self._agent_id = None
            self._cycle_id = None
            report.ended_at = datetime.now()
            report.total_duration_seconds = time.time() - cycle_start
            report.total_llm_calls = sum(p.llm_calls for p in report.phases)
            self._last_report = report

        return report

    def resume_cycle(self, agent_id: Optional[str] = None) -> Optional[DreamReport]:
        """Find and resume the most recent incomplete dream cycle.

        Returns:
            DreamReport if an incomplete cycle was found and resumed, else None.
        """
        cycle_id = self._cortex.graph.get_last_incomplete_cycle(agent_id)
        if not cycle_id:
            logger.info("No incomplete dream cycle to resume")
            return None
        logger.info(f"Resuming incomplete cycle: {cycle_id}")
        return self.run_cycle(agent_id=agent_id, cycle_id=cycle_id)

    def run_all_agents_cycle(self) -> list[DreamReport]:
        """Run a scoped dream cycle for each known agent.

        Auto-discovers agents from the registry, falling back to DISTINCT
        agent_id from memory_nodes. Shared memories participate in every
        agent's dream cycle.

        Returns:
            List of DreamReport, one per agent.
        """
        agents = self._cortex.graph.list_agents()
        if agents:
            agent_ids = [a.id for a in agents]
        else:
            rows = self._cortex.graph.conn.execute(
                "SELECT DISTINCT agent_id FROM memory_nodes"
            ).fetchall()
            agent_ids = [r["agent_id"] for r in rows]

        if not agent_ids:
            return [self.run_cycle()]

        reports = []
        for aid in agent_ids:
            reports.append(self.run_cycle(agent_id=aid))
        self._last_reports = reports
        return reports

    def _notify_phase(self, phase_report: PhaseReport) -> None:
        """Invoke the on_phase_complete callback if registered."""
        if self._on_phase_complete:
            try:
                self._on_phase_complete(phase_report, self._agent_id)
            except Exception as e:
                logger.warning(f"Phase callback error: {e}")

    # =========================================================================
    # Phase 1: SWS Replay
    # =========================================================================

    def _phase_sws_replay(self) -> PhaseReport:
        """Slow-Wave Sleep: replay recent episodes, strengthen temporal links.

        Algorithmic phase - no LLM needed.
        - Replays each unconsolidated episode
        - Spreads activation from episode memories
        - Strengthens co-activated links (Hebbian)
        """
        report = PhaseReport(phase=DreamPhase.SWS_REPLAY)
        start = time.time()

        try:
            episodes = self._cortex.episodes.get_unconsolidated(agent_id=self._agent_id)
            if not episodes:
                report.notes = "No unconsolidated episodes"
                report.duration_seconds = time.time() - start
                self._log_phase(report)
                return report

            total_strengthened = 0
            total_processed = 0

            for ep in episodes:
                # Get memory IDs in this episode
                mem_ids = self._cortex.episodes.get_episode_memories(ep.id)
                if len(mem_ids) < 2:
                    continue

                total_processed += len(mem_ids)

                # Spread activation from episode memories
                seed_weights = [0.9] * len(mem_ids)
                activated = self._cortex.links.spread_activation(
                    seed_ids=mem_ids,
                    seed_weights=seed_weights,
                    max_hops=1,
                    agent_id=self._agent_id,
                )

                # Strengthen co-activated links (Hebbian learning)
                activated_ids = list(activated.keys())
                if len(activated_ids) > 1:
                    n = self._cortex.links.strengthen_co_activated(
                        activated_ids, boost=0.08,
                    )
                    total_strengthened += n

            report.memories_processed = total_processed
            report.links_strengthened = total_strengthened
            report.notes = f"Replayed {len(episodes)} episodes"

        except Exception as e:
            logger.error(f"SWS Replay failed: {e}", exc_info=True)
            report.success = False
            report.notes = str(e)

        report.duration_seconds = time.time() - start
        self._log_phase(report)
        return report

    # =========================================================================
    # Phase 2: Pattern Extraction
    # =========================================================================

    def _phase_pattern_extraction(self) -> PhaseReport:
        """Extract reusable procedures from clusters of similar memories.

        LLM-assisted: groups memories by shared concepts/tags, asks LLM
        to extract actionable procedures.
        """
        report = PhaseReport(phase=DreamPhase.PATTERN_EXTRACTION)
        start = time.time()

        try:
            # Cluster memories by shared tags/concepts
            clusters = self._cluster_by_concepts()

            if not clusters:
                report.notes = "No clusters found for pattern extraction"
                report.duration_seconds = time.time() - start
                self._log_phase(report)
                return report

            total_procedures = 0
            phase_budget = min(DREAM_LLM_BUDGET_PATTERN, self._llm_calls_remaining)

            for concept, mem_ids in clusters.items():
                if phase_budget <= 0 or self._llm_calls_remaining <= 0:
                    break

                # Gather memory contents
                memories_text = self._format_memories_for_llm(mem_ids[:10])
                if not memories_text:
                    continue

                # LLM: extract patterns
                patterns, called = self._llm_extract_patterns(memories_text)
                if called:
                    report.llm_calls += 1
                    phase_budget -= 1
                report.memories_processed += len(mem_ids)

                # Store extracted procedures (with dedup for idempotency on resume)
                for pattern in patterns:
                    content = pattern.get("content") or pattern.get("pattern") or pattern.get("procedure")
                    if not content:
                        continue

                    # Idempotency: skip if a procedure with identical content exists
                    if self._cortex.graph.find_duplicate_content(content):
                        continue

                    source_indices = pattern.get("source_indices", [])
                    source_ids = [mem_ids[i] for i in source_indices if i < len(mem_ids)]

                    proc = self._cortex.procedural.store_procedure(
                        content=content,
                        tags=pattern.get("tags", [concept]),
                        derived_from=source_ids or mem_ids[:3],
                        agent_id=self._agent_id or "CLAUDE",
                    )
                    if proc:
                        total_procedures += 1
                        report.links_created += len(source_ids) or 3

            report.procedures_extracted = total_procedures
            report.notes = f"Extracted {total_procedures} procedures from {len(clusters)} clusters"

        except Exception as e:
            logger.error(f"Pattern Extraction failed: {e}", exc_info=True)
            report.success = False
            report.notes = str(e)

        report.duration_seconds = time.time() - start
        self._log_phase(report)
        return report

    # =========================================================================
    # Phase 3: Schema Formation
    # =========================================================================

    def _phase_schema_formation(self) -> PhaseReport:
        """Abstract episodes into general principles (schemas).

        LLM-assisted: for each episode with enough steps, asks LLM
        to extract a general principle.
        """
        report = PhaseReport(phase=DreamPhase.SCHEMA_FORMATION)
        start = time.time()

        try:
            episodes = self._cortex.episodes.get_unconsolidated(agent_id=self._agent_id)
            total_schemas = 0
            phase_budget = min(DREAM_LLM_BUDGET_SCHEMA, self._llm_calls_remaining)

            for ep in episodes:
                if phase_budget <= 0 or self._llm_calls_remaining <= 0:
                    break

                mem_ids = self._cortex.episodes.get_episode_memories(ep.id)
                if len(mem_ids) < 2:
                    continue

                # Check if schema already exists for these memories
                episode_node = self._cortex.graph.get_node(mem_ids[0]) if mem_ids else None
                existing_tags = episode_node.metadata.tags if episode_node else []
                existing = self._cortex.schemas.find_matching_schemas(tags=existing_tags)
                if existing:
                    # Reinforce existing schema instead
                    for mid in mem_ids:
                        self._cortex.schemas.reinforce_schema(existing[0].id, mid)
                    report.links_created += len(mem_ids)
                    continue

                # Gather memory contents for LLM
                memories_text = self._format_memories_for_llm(mem_ids)
                if not memories_text:
                    continue

                # LLM: form schema
                schema_data, called = self._llm_form_schema(memories_text)
                if called:
                    report.llm_calls += 1
                    phase_budget -= 1

                if schema_data and schema_data.get("content"):
                    episode_tags = list(set(
                        existing_tags + schema_data.get("tags", [])
                    ))
                    schema = self._cortex.schemas.create_schema(
                        content=schema_data["content"],
                        source_ids=mem_ids,
                        tags=episode_tags,
                        agent_id=self._agent_id or "CLAUDE",
                    )
                    if schema:
                        total_schemas += 1
                        report.links_created += len(mem_ids)

                report.memories_processed += len(mem_ids)

            # Evaluate existing schemas for promotion/demotion
            eval_result = self._cortex.schemas.evaluate_schema_candidates()
            promoted = eval_result.get("promoted", 0)
            demoted = eval_result.get("demoted", 0)
            report.memories_pruned += demoted

            report.schemas_extracted = total_schemas
            report.notes = (
                f"Formed {total_schemas} schemas from {len(episodes)} episodes"
                f", promoted {promoted}, demoted {demoted}"
            )

        except Exception as e:
            logger.error(f"Schema Formation failed: {e}", exc_info=True)
            report.success = False
            report.notes = str(e)

        report.duration_seconds = time.time() - start
        self._log_phase(report)
        return report

    # =========================================================================
    # Phase 4: Emotional Reprocessing
    # =========================================================================

    def _phase_emotional_reprocessing(self) -> PhaseReport:
        """Adjust salience based on outcomes. Negative outcomes get higher salience.

        Algorithmic phase (no LLM).
        - Re-analyze emotion for unconsolidated episode memories
        - Boost salience for negative outcomes (learn from mistakes)
        """
        report = PhaseReport(phase=DreamPhase.EMOTIONAL_REPROCESSING)
        start = time.time()

        try:
            episodes = self._cortex.episodes.get_unconsolidated(agent_id=self._agent_id)

            for ep in episodes:
                mem_ids = self._cortex.episodes.get_episode_memories(ep.id)

                for mid in mem_ids:
                    node = self._cortex.graph.get_node(mid)
                    if not node:
                        continue

                    # Re-analyze emotion from current content
                    valence, arousal, sal_adj = self._cortex.affect.analyze_emotion(
                        node.content,
                    )

                    # Reprocess with episode's overall valence
                    self._cortex.affect.reprocess_emotion(
                        node_id=mid,
                        outcome=ep.overall_valence,
                        salience_boost=sal_adj,
                    )
                    report.memories_processed += 1

            report.notes = f"Reprocessed emotions for {report.memories_processed} memories"

        except Exception as e:
            logger.error(f"Emotional Reprocessing failed: {e}", exc_info=True)
            report.success = False
            report.notes = str(e)

        report.duration_seconds = time.time() - start
        self._log_phase(report)
        return report

    # =========================================================================
    # Phase 5: Pruning
    # =========================================================================

    def _phase_pruning(self) -> PhaseReport:
        """Synaptic homeostasis: decay, promote, and prune memories.

        Algorithmic phase:
        - Run decay sweep (recompute activations)
        - Run promotion sweep (move worthy memories up layers)
        - Delete isolated, low-salience sensory memories past age threshold
        """
        report = PhaseReport(phase=DreamPhase.PRUNING)
        start = time.time()

        try:
            # Decay sweep
            decayed = self._cortex.executive.run_decay_sweep()

            # Promotion sweep
            promotions = self._cortex.executive.run_promotion_sweep()
            promo_count = sum(promotions.values()) if promotions else 0

            # Prune old, low-salience, isolated memories
            pruned = 0
            cutoff = datetime.now() - timedelta(hours=DREAM_PRUNING_MIN_AGE_HOURS)
            all_ids = self._cortex.graph.get_all_node_ids(agent_id=self._agent_id)

            for mid in all_ids:
                node = self._cortex.graph.get_node(mid)
                if not node:
                    continue

                # Only prune memories owned by this agent (don't prune shared from other agents)
                if self._agent_id and node.metadata.agent_id != self._agent_id:
                    continue

                # Only prune sensory layer, low salience
                if node.metadata.layer.value != "sensory":
                    continue
                if node.metadata.salience > DREAM_PRUNING_MAX_SALIENCE:
                    continue
                if node.created_at > cutoff:
                    continue

                # Check isolation (no links)
                degree = self._cortex.graph.get_degree(mid)
                if degree > 0:
                    continue

                # Safe to prune
                self._cortex.graph.delete_node(mid)
                pruned += 1

            report.memories_processed = decayed
            report.memories_pruned = pruned
            report.notes = (
                f"Decayed {decayed} memories, "
                f"promoted {promo_count}, "
                f"pruned {pruned} isolated sensory memories"
            )

        except Exception as e:
            logger.error(f"Pruning failed: {e}", exc_info=True)
            report.success = False
            report.notes = str(e)

        report.duration_seconds = time.time() - start
        self._log_phase(report)
        return report

    # =========================================================================
    # Phase 6: REM Recombination
    # =========================================================================

    def _phase_rem_recombination(self) -> PhaseReport:
        """REM dreaming: sample diverse memories, find unexpected connections.

        LLM-assisted: picks random memory pairs from different types/contexts,
        asks LLM if there's a meaningful connection.
        """
        report = PhaseReport(phase=DreamPhase.REM_RECOMBINATION)
        start = time.time()

        try:
            all_ids = self._cortex.graph.get_all_node_ids(agent_id=self._agent_id)
            if len(all_ids) < 4:
                report.notes = "Not enough memories for REM recombination"
                report.duration_seconds = time.time() - start
                self._log_phase(report)
                return report

            # Sample diverse memories
            sample_size = min(DREAM_REM_SAMPLE_SIZE, len(all_ids))
            sample = random.sample(all_ids, sample_size)

            # Load content
            sample_nodes = {}
            for mid in sample:
                node = self._cortex.graph.get_node(mid)
                if node and len(node.content) > 20:
                    sample_nodes[mid] = node

            # Generate random pairs from different types
            pairs_checked = 0
            links_created = 0
            node_list = list(sample_nodes.items())

            phase_budget = min(DREAM_LLM_BUDGET_REM, self._llm_calls_remaining)

            for _ in range(min(DREAM_REM_PAIR_CHECKS, len(node_list) * (len(node_list) - 1) // 2)):
                if phase_budget <= 0 or self._llm_calls_remaining <= 0:
                    break
                if len(node_list) < 2:
                    break

                # Pick two random different memories
                idx_a, idx_b = random.sample(range(len(node_list)), 2)
                id_a, node_a = node_list[idx_a]
                id_b, node_b = node_list[idx_b]

                # Skip if already connected
                if self._cortex.graph.has_link(id_a, id_b):
                    continue

                # Prefer pairs from different types for creative connections
                if node_a.metadata.memory_type == node_b.metadata.memory_type:
                    if random.random() > 0.3:  # 70% skip same-type pairs
                        continue

                # LLM: check for connection
                connection, called = self._llm_rem_connect(node_a.content, node_b.content)
                if called:
                    report.llm_calls += 1
                    phase_budget -= 1
                pairs_checked += 1

                if connection and connection.get("connected"):
                    weight = min(max(connection.get("weight", 0.4), 0.1), 0.9)
                    if weight >= DREAM_REM_MIN_CONNECTION_STRENGTH:
                        link_type_str = connection.get("link_type", "semantic")
                        try:
                            lt = LinkType(link_type_str)
                        except ValueError:
                            lt = LinkType.SEMANTIC

                        self._cortex.links.create_link(
                            source_id=id_a,
                            target_id=id_b,
                            link_type=lt,
                            weight=weight,
                            source="dream_rem",
                            evidence=connection.get("reason", "REM recombination"),
                        )
                        links_created += 1

            report.memories_processed = len(sample_nodes)
            report.links_created = links_created
            report.notes = f"Checked {pairs_checked} pairs, created {links_created} new connections"

        except Exception as e:
            logger.error(f"REM Recombination failed: {e}", exc_info=True)
            report.success = False
            report.notes = str(e)

        report.duration_seconds = time.time() - start
        self._log_phase(report)
        return report

    # =========================================================================
    # LLM helpers
    # =========================================================================

    def _llm_call(self, prompt: str, system: str = SYSTEM_DREAM) -> tuple[Optional[str], bool]:
        """Make an LLM call with budget tracking.

        Returns:
            Tuple of (response_text or None, was_actually_called).
        """
        if not self._llm:
            return None, False
        if self._llm_calls_remaining <= 0:
            logger.warning("LLM call budget exhausted")
            return None, False

        self._llm_calls_remaining -= 1
        try:
            resp = self._llm.generate(prompt=prompt, system=system)
            return resp.text, True
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None, True  # Was called, but failed

    def _parse_json(self, text: Optional[str]) -> Optional[any]:
        """Parse JSON from LLM response, handling markdown fences and preamble text."""
        if not text:
            return None
        cleaned = text.strip()

        # Strip markdown code fences
        if "```" in cleaned:
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Find JSON object or array in mixed text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = cleaned.find(start_char)
            if start == -1:
                continue
            end = cleaned.rfind(end_char)
            if end <= start:
                continue
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue

        logger.warning(f"Failed to parse LLM JSON: {cleaned[:100]}...")
        return None

    def _llm_extract_patterns(self, memories_text: str) -> tuple[list[dict], bool]:
        """Ask LLM to extract patterns from memory cluster.

        Returns: (patterns_list, was_called)
        """
        prompt = PROMPT_EXTRACT_PATTERNS.format(memories=memories_text)
        raw, called = self._llm_call(prompt)
        parsed = self._parse_json(raw)
        if isinstance(parsed, list):
            return parsed, called
        return [], called

    def _llm_form_schema(self, memories_text: str) -> tuple[Optional[dict], bool]:
        """Ask LLM to form an abstract schema from memories.

        Returns: (schema_dict_or_None, was_called)
        """
        prompt = PROMPT_FORM_SCHEMA.format(memories=memories_text)
        raw, called = self._llm_call(prompt)
        parsed = self._parse_json(raw) if raw else None
        return parsed, called

    def _llm_rem_connect(self, content_a: str, content_b: str) -> tuple[Optional[dict], bool]:
        """Ask LLM to find unexpected connection between two memories.

        Returns: (connection_dict_or_None, was_called)
        """
        prompt = PROMPT_REM_CONNECT.format(
            memory_a=content_a[:300],
            memory_b=content_b[:300],
        )
        raw, called = self._llm_call(prompt)
        parsed = self._parse_json(raw) if raw else None
        return parsed, called

    # =========================================================================
    # Clustering helpers
    # =========================================================================

    def _cluster_by_concepts(self) -> dict[str, list[str]]:
        """Cluster memories by shared concepts/tags.

        Returns dict: concept -> [memory_id, ...]
        Only returns clusters meeting minimum size.
        """
        concept_map: dict[str, list[str]] = {}
        all_ids = self._cortex.graph.get_all_node_ids(agent_id=self._agent_id)

        for mid in all_ids:
            node = self._cortex.graph.get_node(mid)
            if not node:
                continue

            # Use both concepts and tags as clustering keys
            keys = set(node.metadata.concepts[:5] + node.metadata.tags)
            for key in keys:
                if key not in concept_map:
                    concept_map[key] = []
                concept_map[key].append(mid)

        # Filter to clusters meeting minimum size
        return {
            k: v for k, v in concept_map.items()
            if len(v) >= DREAM_CLUSTER_MIN_SIZE
        }

    def _format_memories_for_llm(self, mem_ids: list[str]) -> str:
        """Format memory contents for LLM prompt."""
        lines = []
        for i, mid in enumerate(mem_ids[:10]):
            node = self._cortex.graph.get_node(mid)
            if node:
                content = node.content[:200]
                lines.append(f"[{i}] {content}")
        return "\n".join(lines)

    # =========================================================================
    # Logging
    # =========================================================================

    def _log_phase(self, report: PhaseReport) -> None:
        """Log a dream phase to SQLite with cycle_id for checkpointing."""
        try:
            self._cortex.graph.log_dream_phase(
                phase=report.phase.value,
                memories_processed=report.memories_processed,
                links_created=report.links_created,
                links_strengthened=report.links_strengthened,
                memories_pruned=report.memories_pruned,
                schemas_extracted=report.schemas_extracted,
                notes=report.notes,
                success=report.success,
                cycle_id=self._cycle_id,
                agent_id=self._agent_id,
            )
        except Exception as e:
            logger.error(f"Failed to log dream phase: {e}")
