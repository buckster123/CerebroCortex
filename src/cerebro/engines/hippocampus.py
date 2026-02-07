"""EpisodicEngine - Episode management and temporal binding.

The hippocampus: records temporal sequences of events, manages episode
boundaries, creates temporal links between sequential memories.

Responsibilities:
- Episode lifecycle (start, add step, end)
- Temporal link creation between sequential memories
- Session boundary detection
- Episode retrieval by theme or time
"""

from datetime import datetime
from typing import Optional

from cerebro.models.episode import Episode, EpisodeStep
from cerebro.models.memory import MemoryNode
from cerebro.storage.graph_store import GraphStore
from cerebro.types import EmotionalValence, LinkType


class EpisodicEngine:
    """Manages episodic memory: temporal sequences and narrative structure."""

    def __init__(self, graph: GraphStore):
        self._graph = graph
        self._active_episodes: dict[str, Episode] = {}  # session_id -> Episode

    def start_episode(
        self,
        title: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: str = "CLAUDE",
        tags: Optional[list[str]] = None,
    ) -> Episode:
        """Start recording a new episode.

        Args:
            title: Optional descriptive title
            session_id: Session this episode belongs to
            agent_id: Agent recording the episode
            tags: Tags for the episode

        Returns:
            The new Episode instance
        """
        episode = Episode(
            title=title,
            session_id=session_id,
            agent_id=agent_id,
            tags=tags or [],
            started_at=datetime.now(),
        )
        self._graph.add_episode(episode)

        # Track as active if session-bound
        if session_id:
            self._active_episodes[session_id] = episode

        return episode

    def add_step(
        self,
        episode_id: str,
        memory_id: str,
        role: str = "event",
    ) -> Optional[EpisodeStep]:
        """Add a memory as a step in an episode.

        Also creates temporal links from the previous step to this one.

        Args:
            episode_id: Episode to add to
            memory_id: Memory being added as a step
            role: Role of this step (event, context, outcome, reflection)

        Returns:
            The created EpisodeStep, or None if episode not found.
        """
        episode = self._graph.get_episode(episode_id)
        if not episode:
            return None

        # Determine position
        position = len(episode.steps)

        step = EpisodeStep(
            memory_id=memory_id,
            position=position,
            role=role,
        )
        self._graph.add_episode_step(episode_id, step)

        # Create temporal link from previous step
        if episode.steps:
            prev_step = episode.steps[-1]
            self._graph.ensure_link(
                prev_step.memory_id, memory_id, LinkType.TEMPORAL,
                weight=0.7, source="encoding",
                evidence=f"Sequential in episode {episode_id}",
            )

        # Create part_of link to mark episode membership
        # (Using episode_id in metadata instead since episodes aren't memory nodes)
        self._graph.update_node_metadata(memory_id, episode_id=episode_id)

        return step

    def end_episode(
        self,
        episode_id: str,
        summary: Optional[str] = None,
        valence: EmotionalValence = EmotionalValence.NEUTRAL,
        peak_arousal: float = 0.5,
    ) -> Optional[Episode]:
        """End an episode, setting its end time and emotional summary.

        Args:
            episode_id: Episode to end
            summary: Optional summary text (could become a schematic memory)
            valence: Overall emotional valence of the episode
            peak_arousal: Peak emotional arousal during the episode

        Returns:
            The updated Episode, or None if not found.
        """
        # Update episode in SQLite
        self._graph.conn.execute(
            """UPDATE episodes SET
                ended_at = ?, overall_valence = ?, peak_arousal = ?,
                title = COALESCE(?, title)
            WHERE id = ?""",
            (
                datetime.now().isoformat(), valence.value, peak_arousal,
                summary, episode_id,
            ),
        )
        self._graph.conn.commit()

        # Remove from active episodes
        for session_id, ep in list(self._active_episodes.items()):
            if ep.id == episode_id:
                del self._active_episodes[session_id]
                break

        return self._graph.get_episode(episode_id)

    def get_active_episode(self, session_id: str) -> Optional[Episode]:
        """Get the currently active episode for a session."""
        if session_id in self._active_episodes:
            # Refresh from database
            ep = self._active_episodes[session_id]
            return self._graph.get_episode(ep.id)
        return None

    def add_to_current_episode(
        self,
        session_id: str,
        memory_id: str,
        role: str = "event",
    ) -> Optional[EpisodeStep]:
        """Add a memory to the current session's active episode.

        If no active episode exists for this session, creates one.
        """
        if session_id not in self._active_episodes:
            episode = self.start_episode(session_id=session_id)
        else:
            episode = self._active_episodes[session_id]

        return self.add_step(episode.id, memory_id, role)

    def get_episode_memories(self, episode_id: str) -> list[str]:
        """Get all memory IDs in an episode, ordered by position."""
        episode = self._graph.get_episode(episode_id)
        if not episode:
            return []
        return [step.memory_id for step in sorted(episode.steps, key=lambda s: s.position)]

    def get_unconsolidated(self, agent_id: Optional[str] = None) -> list[Episode]:
        """Get episodes not yet processed by the Dream Engine."""
        return self._graph.get_unconsolidated_episodes(agent_id=agent_id)

    def mark_consolidated(self, episode_id: str) -> bool:
        """Mark an episode as processed by the Dream Engine."""
        cursor = self._graph.conn.execute(
            "UPDATE episodes SET consolidated = 1 WHERE id = ?",
            (episode_id,),
        )
        self._graph.conn.commit()
        return cursor.rowcount > 0

    def get_recent_episodes(
        self,
        limit: int = 10,
        agent_id: Optional[str] = None,
    ) -> list[Episode]:
        """Get recently created episodes."""
        if agent_id:
            rows = self._graph.conn.execute(
                "SELECT id FROM episodes WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        else:
            rows = self._graph.conn.execute(
                "SELECT id FROM episodes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [ep for r in rows if (ep := self._graph.get_episode(r["id"]))]
