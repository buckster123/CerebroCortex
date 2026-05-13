"""SQLite schema for CerebroCortex graph persistence.

This is the canonical store for the associative graph, full metadata,
and strength parameters. ChromaDB handles vectors; SQLite handles everything else.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 6

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
    recipient TEXT,

    -- Source tracking
    source TEXT NOT NULL DEFAULT 'user_input',
    derived_from_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    promoted_at TEXT,

    -- Multimodal support (Phase B)
    media_type TEXT NOT NULL DEFAULT 'text',
    source_file TEXT,

    -- CRUD lifecycle (Phase C)
    deleted_at TEXT
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

-- Attachments (multimodal media linked to memories)
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    media_type TEXT NOT NULL DEFAULT 'unknown',
    file_path TEXT,
    original_bytes_hash TEXT,
    text_description TEXT,
    vision_embedding_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attachments_memory ON attachments(memory_id);
CREATE INDEX IF NOT EXISTS idx_attachments_media_type ON attachments(media_type);

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
    cycle_id TEXT,
    agent_id TEXT,
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

CREATE INDEX IF NOT EXISTS idx_nodes_source ON memory_nodes(source);

-- Full-text search (FTS5) virtual table for keyword fallback
CREATE VIRTUAL TABLE IF NOT EXISTS memory_nodes_fts USING fts5(
    content,
    content_rowid=rowid,
    content=memory_nodes
);

-- Triggers to keep FTS5 in sync with memory_nodes
CREATE TRIGGER IF NOT EXISTS memory_nodes_fts_insert
AFTER INSERT ON memory_nodes BEGIN
    INSERT INTO memory_nodes_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_nodes_fts_update
AFTER UPDATE OF content ON memory_nodes BEGIN
    INSERT INTO memory_nodes_fts(memory_nodes_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO memory_nodes_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_nodes_fts_delete
AFTER DELETE ON memory_nodes BEGIN
    INSERT INTO memory_nodes_fts(memory_nodes_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;

-- Memory versions (audit trail for content changes)
CREATE TABLE IF NOT EXISTS memory_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    content TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    salience REAL NOT NULL,
    visibility TEXT NOT NULL,
    edited_by TEXT,
    edited_at TEXT NOT NULL,
    change_note TEXT,
    FOREIGN KEY (memory_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_versions_memory ON memory_versions(memory_id);

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

    # Check FTS5 availability
    fts5_available = conn.execute(
        "SELECT count(*) FROM pragma_compile_options() WHERE compile_options LIKE 'ENABLE_FTS5'"
    ).fetchone()[0]
    if not fts5_available:
        logger.warning("SQLite FTS5 not available; keyword search fallback disabled")

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

    # Migration v3: ensure recipient column + indexes exist.
    # Runs idempotently on both fresh and existing databases.
    try:
        conn.execute("ALTER TABLE memory_nodes ADD COLUMN recipient TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists (fresh DB or already migrated)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_recipient ON memory_nodes(recipient)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_source ON memory_nodes(source)")
    existing_v3 = conn.execute(
        "SELECT version FROM schema_version WHERE version = 3"
    ).fetchone()
    if not existing_v3:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at, description) "
            "VALUES (3, datetime('now'), 'Add recipient column for agent messaging')"
        )
    conn.commit()

    # Migration v4: add cycle_id + agent_id to dream_log for checkpointing
    try:
        conn.execute("ALTER TABLE dream_log ADD COLUMN cycle_id TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE dream_log ADD COLUMN agent_id TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dream_cycle ON dream_log(cycle_id)")
    existing_v4 = conn.execute(
        "SELECT version FROM schema_version WHERE version = 4"
    ).fetchone()
    if not existing_v4:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at, description) "
            "VALUES (4, datetime('now'), 'Add dream checkpointing columns')"
        )
    conn.commit()

    # Migration v5: multimodal support (attachments table, media_type + source_file columns)
    # The attachments table is handled by CREATE TABLE IF NOT EXISTS above.
    # Existing memory_nodes need the new columns via ALTER TABLE.
    try:
        conn.execute("ALTER TABLE memory_nodes ADD COLUMN media_type TEXT NOT NULL DEFAULT 'text'")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE memory_nodes ADD COLUMN source_file TEXT")
    except sqlite3.OperationalError:
        pass
    existing_v5 = conn.execute(
        "SELECT version FROM schema_version WHERE version = 5"
    ).fetchone()
    if not existing_v5:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at, description) "
            "VALUES (5, datetime('now'), 'Add multimodal attachments and media_type')"
        )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_media_type ON memory_nodes(media_type)")
    conn.commit()

    # Migration v6: soft-delete support (deleted_at column) + memory_versions table
    try:
        conn.execute("ALTER TABLE memory_nodes ADD COLUMN deleted_at TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_deleted ON memory_nodes(deleted_at) WHERE deleted_at IS NOT NULL")
    existing_v6 = conn.execute(
        "SELECT version FROM schema_version WHERE version = 6"
    ).fetchone()
    if not existing_v6:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at, description) "
            "VALUES (6, datetime('now'), 'Add soft-delete and memory versioning')"
        )
    conn.commit()

    return conn
