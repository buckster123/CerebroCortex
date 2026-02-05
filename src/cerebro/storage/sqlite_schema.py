"""SQLite schema for CerebroCortex graph persistence.

This is the canonical store for the associative graph, full metadata,
and strength parameters. ChromaDB handles vectors; SQLite handles everything else.
"""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2

SCHEMA_SQL = """
-- Memory nodes (rich metadata + strength state)
CREATE TABLE IF NOT EXISTS memory_nodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'semantic',
    layer TEXT NOT NULL DEFAULT 'working',
    agent_id TEXT NOT NULL DEFAULT 'CLAUDE',
    visibility TEXT NOT NULL DEFAULT 'shared',

    -- FSRS strength parameters
    stability REAL NOT NULL DEFAULT 1.0,
    difficulty REAL NOT NULL DEFAULT 5.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    access_timestamps_json TEXT NOT NULL DEFAULT '[]',
    compressed_count INTEGER NOT NULL DEFAULT 0,
    compressed_avg_interval REAL NOT NULL DEFAULT 0.0,
    last_retrievability REAL NOT NULL DEFAULT 1.0,
    last_activation REAL NOT NULL DEFAULT 0.0,
    last_computed_at REAL,

    -- Emotional dimension
    valence TEXT NOT NULL DEFAULT 'neutral',
    arousal REAL NOT NULL DEFAULT 0.5,
    salience REAL NOT NULL DEFAULT 0.5,

    -- Context
    episode_id TEXT,
    session_id TEXT,
    conversation_thread TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    concepts_json TEXT NOT NULL DEFAULT '[]',
    responding_to_json TEXT NOT NULL DEFAULT '[]',
    related_agents_json TEXT NOT NULL DEFAULT '[]',

    -- Source tracking
    source TEXT NOT NULL DEFAULT 'user_input',
    derived_from_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    promoted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON memory_nodes(memory_type);
CREATE INDEX IF NOT EXISTS idx_nodes_layer ON memory_nodes(layer);
CREATE INDEX IF NOT EXISTS idx_nodes_agent ON memory_nodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_visibility ON memory_nodes(visibility);
CREATE INDEX IF NOT EXISTS idx_nodes_episode ON memory_nodes(episode_id);
CREATE INDEX IF NOT EXISTS idx_nodes_session ON memory_nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_nodes_salience ON memory_nodes(salience DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_created ON memory_nodes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_content_hash ON memory_nodes(content_hash);

-- Associative links (graph edges)
CREATE TABLE IF NOT EXISTS associative_links (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    link_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,

    activation_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_activated TEXT,

    source_reason TEXT NOT NULL DEFAULT 'system',
    evidence TEXT,

    FOREIGN KEY (source_id) REFERENCES memory_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memory_nodes(id) ON DELETE CASCADE,
    UNIQUE(source_id, target_id, link_type)
);

CREATE INDEX IF NOT EXISTS idx_links_source ON associative_links(source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON associative_links(target_id);
CREATE INDEX IF NOT EXISTS idx_links_type ON associative_links(link_type);
CREATE INDEX IF NOT EXISTS idx_links_weight ON associative_links(weight DESC);

-- Episodes (temporal sequences)
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    title TEXT,
    agent_id TEXT NOT NULL DEFAULT 'CLAUDE',
    session_id TEXT,

    started_at TEXT,
    ended_at TEXT,

    overall_valence TEXT NOT NULL DEFAULT 'neutral',
    peak_arousal REAL NOT NULL DEFAULT 0.5,
    tags_json TEXT NOT NULL DEFAULT '[]',

    consolidated INTEGER NOT NULL DEFAULT 0,
    schema_extracted INTEGER NOT NULL DEFAULT 0,

    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated);

-- Episode steps (ordered memories within an episode)
CREATE TABLE IF NOT EXISTS episode_steps (
    episode_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'event',
    timestamp TEXT NOT NULL,

    PRIMARY KEY (episode_id, position),
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
    FOREIGN KEY (memory_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
);

-- Agent profiles
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
    lineage TEXT,
    specialization TEXT,
    origin_story TEXT,
    color TEXT DEFAULT '#888888',
    symbol TEXT DEFAULT 'A',
    registered_at TEXT NOT NULL
);

-- Dream engine log
CREATE TABLE IF NOT EXISTS dream_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    memories_processed INTEGER DEFAULT 0,
    links_created INTEGER DEFAULT 0,
    links_strengthened INTEGER DEFAULT 0,
    memories_pruned INTEGER DEFAULT 0,
    schemas_extracted INTEGER DEFAULT 0,
    notes TEXT,
    success INTEGER NOT NULL DEFAULT 1
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT
);
"""


def initialize_database(db_path: Path) -> sqlite3.Connection:
    """Create or open the SQLite database with full schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    # Apply schema
    conn.executescript(SCHEMA_SQL)

    # Record schema version
    existing = conn.execute(
        "SELECT version FROM schema_version WHERE version = ?", (SCHEMA_VERSION,)
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at, description) VALUES (?, datetime('now'), ?)",
            (SCHEMA_VERSION, "Initial CerebroCortex schema"),
        )
        conn.commit()

    return conn
