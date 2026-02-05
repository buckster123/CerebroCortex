"""Import memories from Neo-Cortex export JSON into CerebroCortex.

Neo-Cortex exports use the MemoryCore format:
{
    "format_version": "1.0",
    "agent_id": "CLAUDE",
    "exported_at": "2026-...",
    "collections": {
        "cortex_shared": [ { record }, ... ],
        "cortex_private": [ ... ],
        ...
    },
    "metadata": { ... }
}

This importer:
1. Maps Neo-Cortex message_type -> CerebroCortex memory_type
2. Maps attention_weight -> salience
3. Seeds ACT-R strength from access_count + created_at
4. Creates contextual links from responding_to
5. Registers agent profiles
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from cerebro.cortex import CerebroCortex
from cerebro.models.agent import AgentProfile
from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.types import (
    EmotionalValence,
    LinkType,
    MemoryLayer,
    MemoryType,
    Visibility,
)


# =========================================================================
# Type mapping
# =========================================================================

MESSAGE_TYPE_MAP: dict[str, MemoryType] = {
    "fact": MemoryType.SEMANTIC,
    "observation": MemoryType.SEMANTIC,
    "discovery": MemoryType.SEMANTIC,
    "cultural": MemoryType.SEMANTIC,
    "dialogue": MemoryType.EPISODIC,
    "question": MemoryType.EPISODIC,
    "clarification": MemoryType.EPISODIC,
    "task": MemoryType.PROSPECTIVE,
    "protocol": MemoryType.PROCEDURAL,
    "reminder": MemoryType.PROSPECTIVE,
    "session_note": MemoryType.EPISODIC,
}

VISIBILITY_MAP: dict[str, Visibility] = {
    "private": Visibility.PRIVATE,
    "shared": Visibility.SHARED,
    "thread": Visibility.THREAD,
}

LAYER_MAP: dict[str, MemoryLayer] = {
    "sensory": MemoryLayer.SENSORY,
    "working": MemoryLayer.WORKING,
    "long_term": MemoryLayer.LONG_TERM,
    "cortex": MemoryLayer.CORTEX,
}


# =========================================================================
# Import result
# =========================================================================


@dataclass
class ImportReport:
    """Report from a Neo-Cortex import operation."""
    memories_imported: int = 0
    memories_skipped: int = 0
    links_created: int = 0
    agents_registered: int = 0
    errors: list[str] = field(default_factory=list)
    id_mapping: dict[str, str] = field(default_factory=dict)  # old_id -> new_id
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "memories_imported": self.memories_imported,
            "memories_skipped": self.memories_skipped,
            "links_created": self.links_created,
            "agents_registered": self.agents_registered,
            "errors": self.errors[:20],  # cap for readability
            "total_errors": len(self.errors),
            "duration_seconds": round(self.duration_seconds, 2),
        }


# =========================================================================
# Importer
# =========================================================================


class NeoCortexImporter:
    """Imports Neo-Cortex export data into CerebroCortex."""

    def __init__(self, cortex: CerebroCortex):
        self.cortex = cortex
        self._report = ImportReport()

    def import_file(self, path: Path) -> ImportReport:
        """Import from a Neo-Cortex JSON export file."""
        with open(path) as f:
            data = json.load(f)
        return self.import_data(data)

    def import_data(self, data: dict) -> ImportReport:
        """Import from parsed Neo-Cortex export dict.

        Two-pass approach:
        1. Create all memory nodes, building an old_id -> new_id mapping
        2. Create links from responding_to relationships
        """
        self._report = ImportReport()
        start = time.time()

        collections = data.get("collections", {})

        # Collect all records across all collections
        all_records: list[tuple[str, dict]] = []
        for collection_name, records in collections.items():
            for record in records:
                all_records.append((collection_name, record))

        # Pass 1: import memories (skip agent_profile, those become agents)
        for collection_name, record in all_records:
            msg_type = record.get("message_type", "observation")

            if msg_type == "agent_profile":
                self._import_agent(record)
                continue

            self._import_memory(record, collection_name)

        # Pass 2: create links from responding_to
        for _coll, record in all_records:
            if record.get("message_type") == "agent_profile":
                continue
            self._create_responding_to_links(record)

        # Resync igraph after bulk insert
        self.cortex.graph.resync_igraph()

        self._report.duration_seconds = time.time() - start
        return self._report

    def _import_memory(self, record: dict, collection: str) -> Optional[str]:
        """Import a single Neo-Cortex record as a CerebroCortex MemoryNode."""
        content = record.get("content", "").strip()
        if not content or len(content) < 3:
            self._report.memories_skipped += 1
            return None

        old_id = record.get("id", "")

        # Check for duplicate content
        existing = self.cortex.graph.find_duplicate_content(content)
        if existing:
            self._report.memories_skipped += 1
            if old_id:
                self._report.id_mapping[old_id] = existing
            return existing

        # Map types
        msg_type = record.get("message_type", "observation")
        memory_type = MESSAGE_TYPE_MAP.get(msg_type, MemoryType.SEMANTIC)
        visibility = VISIBILITY_MAP.get(record.get("visibility", "shared"), Visibility.SHARED)
        layer = LAYER_MAP.get(record.get("layer", "working"), MemoryLayer.WORKING)

        # Map fields
        tags = record.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = [tags]

        related_agents = record.get("related_agents", [])
        if isinstance(related_agents, str):
            try:
                related_agents = json.loads(related_agents)
            except (json.JSONDecodeError, TypeError):
                related_agents = []

        responding_to = record.get("responding_to", [])
        if isinstance(responding_to, str):
            try:
                responding_to = json.loads(responding_to)
            except (json.JSONDecodeError, TypeError):
                responding_to = []

        # Salience from attention_weight (Neo-Cortex default is 1.0, ours is 0.5)
        attention_weight = record.get("attention_weight", 1.0)
        salience = max(0.0, min(1.0, attention_weight))

        # Parse created_at
        created_at = datetime.now()
        raw_created = record.get("created_at")
        if raw_created:
            try:
                created_at = datetime.fromisoformat(str(raw_created))
            except (ValueError, TypeError):
                pass

        # Seed ACT-R strength from access history
        access_count = record.get("access_count", 0) or 0
        strength = self._seed_strength(access_count, created_at)

        # Build metadata
        metadata = MemoryMetadata(
            agent_id=record.get("agent_id", "CLAUDE"),
            visibility=visibility,
            layer=layer,
            memory_type=memory_type,
            tags=tags,
            responding_to=responding_to,
            related_agents=related_agents,
            conversation_thread=record.get("conversation_thread"),
            salience=salience,
            source="import",
        )

        # Add original message_type as tag for provenance
        if msg_type and f"neo:{msg_type}" not in tags:
            metadata.tags = list(metadata.tags) + [f"neo:{msg_type}"]

        # Build node
        node = MemoryNode(
            content=content,
            metadata=metadata,
            strength=strength,
            created_at=created_at,
        )

        try:
            self.cortex.graph.add_node(node)
            self._report.memories_imported += 1
            if old_id:
                self._report.id_mapping[old_id] = node.id
            return node.id
        except Exception as e:
            self._report.errors.append(f"Failed to import {old_id}: {e}")
            self._report.memories_skipped += 1
            return None

    def _create_responding_to_links(self, record: dict) -> None:
        """Create contextual links from responding_to relationships."""
        old_id = record.get("id", "")
        new_id = self._report.id_mapping.get(old_id)
        if not new_id:
            return

        responding_to = record.get("responding_to", [])
        if isinstance(responding_to, str):
            try:
                responding_to = json.loads(responding_to)
            except (json.JSONDecodeError, TypeError):
                return

        for target_old_id in responding_to:
            target_new_id = self._report.id_mapping.get(target_old_id)
            if not target_new_id or target_new_id == new_id:
                continue

            try:
                self.cortex.graph.ensure_link(
                    source_id=new_id,
                    target_id=target_new_id,
                    link_type=LinkType.CONTEXTUAL,
                    weight=0.6,
                    source="migration",
                    evidence="Imported from Neo-Cortex responding_to",
                )
                self._report.links_created += 1
            except Exception as e:
                self._report.errors.append(f"Link {old_id}->{target_old_id}: {e}")

    def _import_agent(self, record: dict) -> None:
        """Extract agent info from an agent_profile record and register."""
        content = record.get("content", "")
        agent_id = record.get("agent_id", "")

        if not agent_id:
            return

        # Parse agent fields from content (Neo-Cortex format)
        display_name = agent_id
        specialization = ""
        generation = 0
        lineage = ""
        symbol = agent_id[0] if agent_id else "A"

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Agent ID:"):
                agent_id = line.split(":", 1)[1].strip()
            elif line.startswith("Display Name:") or line.startswith("Agent Profile:"):
                display_name = line.split(":", 1)[1].strip()
            elif line.startswith("Specialization:"):
                specialization = line.split(":", 1)[1].strip()
            elif line.startswith("Generation:"):
                try:
                    generation = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Lineage:"):
                lineage = line.split(":", 1)[1].strip()

        profile = AgentProfile(
            id=agent_id,
            display_name=display_name,
            generation=generation,
            lineage=lineage,
            specialization=specialization,
            symbol=symbol,
        )

        try:
            self.cortex.graph.register_agent(profile)
            self._report.agents_registered += 1
        except Exception as e:
            self._report.errors.append(f"Agent {agent_id}: {e}")

    @staticmethod
    def _seed_strength(access_count: int, created_at: datetime) -> StrengthState:
        """Seed ACT-R strength parameters from Neo-Cortex access history.

        We spread synthetic access timestamps evenly from creation time to now
        to give the memory a realistic decay curve.
        """
        now = time.time()
        created_ts = created_at.timestamp()
        elapsed = max(1.0, now - created_ts)

        count = max(1, access_count)
        timestamps = []

        # Stability: days since creation, clamped to [0.01, 30.0]
        stability = max(0.01, min(30.0, elapsed / 86400))

        if count <= 50:
            # Generate evenly-spaced timestamps
            interval = elapsed / count
            for i in range(count):
                timestamps.append(created_ts + interval * i)
        else:
            # Keep 50 most recent, compress older
            recent_count = 50
            old_count = count - recent_count
            interval = elapsed / count
            for i in range(count - recent_count, count):
                timestamps.append(created_ts + interval * i)

            return StrengthState(
                stability=stability,
                difficulty=5.0,
                access_timestamps=timestamps,
                access_count=count,
                compressed_count=old_count,
                compressed_avg_interval=interval,
                last_retrievability=0.8,
                last_activation=0.0,
                last_computed_at=now,
            )

        return StrengthState(
            stability=stability,
            difficulty=5.0,
            access_timestamps=timestamps,
            access_count=count,
            last_retrievability=0.8,
            last_activation=0.0,
            last_computed_at=now,
        )
