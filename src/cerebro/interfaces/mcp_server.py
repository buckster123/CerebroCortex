#!/usr/bin/env python3
"""CerebroCortex MCP Server.

Exposes the full CerebroCortex memory system to Claude Code and other MCP clients.

Features:
- remember/recall/associate (core memory operations)
- Episode management (episode_start, episode_add_step, episode_end)
- Session continuity (session_save, session_recall)
- Agent registration
- Memory health and graph stats
- Backward-compatible Neo-Cortex aliases

Usage:
    python -m cerebro.interfaces.mcp_server
    ./cerebro-mcp

Claude Code config (~/.claude.json):
    {
        "mcpServers": {
            "cerebro-cortex": {
                "command": "/path/to/CerebroCore/cerebro-mcp"
            }
        }
    }
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource, ResourceTemplate, Prompt, PromptArgument, PromptMessage, GetPromptResult

from cerebro.config import DEFAULT_AGENT_ID, MCP_SERVER_NAME, MCP_SERVER_VERSION
from cerebro.cortex import CerebroCortex
from cerebro.models.agent import AgentProfile
from cerebro.types import EmotionalValence, LinkType, MemoryType, Visibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("cerebro-mcp")

server = Server(MCP_SERVER_NAME)

_cortex: Optional[CerebroCortex] = None
_dream_engine = None


def get_cortex() -> CerebroCortex:
    global _cortex
    if _cortex is None:
        _cortex = CerebroCortex()
        _cortex.initialize()
    return _cortex


def get_dream_engine(cortex: CerebroCortex):
    global _dream_engine
    if _dream_engine is None:
        from cerebro.engines.dream import DreamEngine
        try:
            from cerebro.utils.llm import LLMClient
            llm = LLMClient()
        except Exception:
            llm = None
        _dream_engine = DreamEngine(cortex, llm_client=llm)
    return _dream_engine


def _coerce_array(val) -> list | None:
    """Handle array params that some MCP bridges serialize as JSON strings.

    Some MCP clients (e.g. Hermes) encode array arguments as JSON strings
    like '["a","b"]' instead of native arrays. This function normalises
    the value so handler code always receives a plain Python list (or None).
    """
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return [val]
    return [val]


# =============================================================================
# Tool Schemas
# =============================================================================

TOOL_SCHEMAS: dict[str, dict] = {
    # -----------------------------------------------------------------
    # Core: remember, recall, associate
    # -----------------------------------------------------------------
    "remember": {
        "description": "Save information to long-term memory. Automatically detects duplicates, categorizes the content, and connects it to related memories. Use this to store facts, decisions, or anything worth remembering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content to store"},
                "memory_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural", "affective", "prospective", "schematic"],
                    "description": "Memory type (auto-classified if omitted)",
                },
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Tags for categorization"},
                "salience": {"type": "number", "description": "Importance 0-1 (auto-estimated if omitted)"},
                "agent_id": {"type": "string", "description": "Agent storing this memory"},
                "session_id": {"type": "string", "description": "Current session ID"},
                "visibility": {
                    "type": "string",
                    "enum": ["private", "shared", "thread"],
                    "description": "Who can see this: 'private' (only you), 'shared' (all agents), 'thread' (same conversation)",
                },
                "context_ids": {
                    "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}],
                    "description": "IDs of related memories to link to",
                },
            },
            "required": ["content"],
        },
    },
    "recall": {
        "description": "Search your memories by meaning, not just keywords. Returns the most relevant memories ranked by relevance, importance, and recency.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {"type": "integer", "description": "Max results (default: 10)"},
                "memory_types": {
                    "anyOf": [{"type": "array", "items": {"type": "string", "enum": ["episodic", "semantic", "procedural", "affective", "prospective", "schematic"]}}, {"type": "string"}],
                    "description": "Filter by memory type",
                },
                "agent_id": {"type": "string", "description": "Filter by agent"},
                "min_salience": {"type": "number", "description": "Minimum importance threshold"},
                "context_ids": {
                    "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}],
                    "description": "IDs of related memories to boost in results",
                },
                "conversation_thread": {"type": "string", "description": "Thread ID for scoping results to a conversation"},
                "explain": {"type": "boolean", "description": "If true, return detailed score breakdown for each result (vector similarity, ACT-R activation, FSRS retrievability, salience)"},
            },
            "required": ["query"],
        },
    },
    "associate": {
        "description": "Create a link between two memories. Links improve search: when one memory is found, linked memories are boosted in results. Link types: temporal, causal, semantic, affective, contextual, contradicts, supports, derived_from, part_of.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source memory ID"},
                "target_id": {"type": "string", "description": "Target memory ID"},
                "link_type": {
                    "type": "string",
                    "enum": ["temporal", "causal", "semantic", "affective", "contextual", "contradicts", "supports", "derived_from", "part_of"],
                    "description": "Type of relationship",
                },
                "weight": {"type": "number", "description": "Link strength 0-1 (default: 0.5)"},
                "evidence": {"type": "string", "description": "Why this link exists"},
            },
            "required": ["source_id", "target_id", "link_type"],
        },
    },

    # -----------------------------------------------------------------
    # Memory CRUD
    # -----------------------------------------------------------------
    "get_memory": {
        "description": "Get a single memory by ID with all its metadata, tags, and concepts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to retrieve"},
                "agent_id": {"type": "string", "description": "Agent ID for access check"},
            },
            "required": ["memory_id"],
        },
    },
    "delete_memory": {
        "description": "Permanently delete a memory. Removes it from both search and the link graph.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to delete"},
                "agent_id": {"type": "string", "description": "Agent ID for access check"},
            },
            "required": ["memory_id"],
        },
    },
    "update_memory": {
        "description": "Update a memory's content, tags, importance, or visibility. Changing content will update its search index automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to update"},
                "content": {"type": "string", "description": "New content (updates search index)"},
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "New tags (replaces existing)"},
                "salience": {"type": "number", "description": "New importance 0-1"},
                "visibility": {
                    "type": "string",
                    "enum": ["private", "shared", "thread"],
                    "description": "Who can see this: 'private' (only you), 'shared' (all agents), 'thread' (same conversation)",
                },
                "agent_id": {"type": "string", "description": "Agent ID for access check"},
            },
            "required": ["memory_id"],
        },
    },

    "share_memory": {
        "description": "Change who can see a memory. Options: 'private' (only the owner), 'shared' (all agents), 'thread' (agents in the same conversation). Only the memory owner can change this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to change"},
                "visibility": {
                    "type": "string",
                    "enum": ["private", "shared", "thread"],
                    "description": "New visibility level",
                },
                "agent_id": {"type": "string", "description": "Requesting agent (must be owner)"},
            },
            "required": ["memory_id", "visibility"],
        },
    },

    # -----------------------------------------------------------------
    # Episodes
    # -----------------------------------------------------------------
    "episode_start": {
        "description": "Start recording a sequence of related events as an episode. Memories added as steps will be linked in order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Episode title"},
                "session_id": {"type": "string", "description": "Session this episode belongs to"},
                "agent_id": {"type": "string", "description": "Agent recording the episode"},
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]}
            },
            "required": [],
        },
    },
    "episode_add_step": {
        "description": "Add a memory as the next step in an episode. Steps are linked in sequence automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string", "description": "Episode to add to"},
                "memory_id": {"type": "string", "description": "Memory being added"},
                "role": {
                    "type": "string",
                    "enum": ["event", "context", "outcome", "reflection"],
                    "description": "Role of this step in the episode",
                },
            },
            "required": ["episode_id", "memory_id"],
        },
    },
    "episode_end": {
        "description": "Finish recording an episode. Optionally add a summary and overall tone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string", "description": "Episode to end"},
                "summary": {"type": "string", "description": "Episode summary"},
                "valence": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "mixed"],
                    "description": "Overall emotional tone",
                },
            },
            "required": ["episode_id"],
        },
    },

    # -----------------------------------------------------------------
    # Episodes & Intentions
    # -----------------------------------------------------------------
    "list_episodes": {
        "description": "List recent episodes with their title, step count, tone, and creation time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max episodes to return (default: 10)"},
                "agent_id": {"type": "string", "description": "Filter by agent"},
            },
            "required": [],
        },
    },
    "get_episode": {
        "description": "Get full details of an episode including all its steps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string", "description": "Episode ID to retrieve"},
                "agent_id": {"type": "string", "description": "Agent ID for access check"},
            },
            "required": ["episode_id"],
        },
    },
    "get_episode_memories": {
        "description": "Get all memories in an episode, in order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string", "description": "Episode to get memories from"},
                "agent_id": {"type": "string", "description": "Agent ID for access check"},
            },
            "required": ["episode_id"],
        },
    },
    "store_intention": {
        "description": "Save a TODO or reminder for future action. The system will surface it when relevant.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The TODO or reminder content"},
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Tags for categorization"},
                "agent_id": {"type": "string", "description": "Agent storing this intention"},
                "salience": {"type": "number", "description": "Importance 0-1 (default: 0.7)"},
            },
            "required": ["content"],
        },
    },
    "list_intentions": {
        "description": "List pending TODOs and reminders that have not been resolved.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Filter by agent"},
                "min_salience": {"type": "number", "description": "Minimum importance threshold (default: 0.3)"},
            },
            "required": [],
        },
    },
    "resolve_intention": {
        "description": "Mark a TODO or reminder as done.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID of the intention to resolve"},
            },
            "required": ["memory_id"],
        },
    },

    # -----------------------------------------------------------------
    # Sessions
    # -----------------------------------------------------------------
    "session_save": {
        "description": "Save a summary of the current session so your future self can pick up where you left off. Include key discoveries, unfinished tasks, and orientation notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_summary": {"type": "string", "description": "What happened this session"},
                "key_discoveries": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Important findings"},
                "unfinished_business": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Tasks to continue"},
                "if_disoriented": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Orientation instructions for future sessions"},
                "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "session_type": {"type": "string", "enum": ["orientation", "technical", "emotional", "task"]},
            },
            "required": ["session_summary"],
        },
    },
    "session_recall": {
        "description": "Retrieve notes from previous sessions for orientation. Useful at the start of a new session to remember where you left off.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lookback_hours": {"type": "integer", "description": "How far back to search (default: 168 = 1 week)"},
                "priority_filter": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "session_type": {"type": "string", "enum": ["orientation", "technical", "emotional", "task"]},
                "limit": {"type": "integer", "description": "Max sessions to return (default: 10)"},
            },
            "required": [],
        },
    },

    # -----------------------------------------------------------------
    # Agents
    # -----------------------------------------------------------------
    "register_agent": {
        "description": "Register a new agent in the memory system. Each agent gets its own memory space with configurable sharing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Unique agent identifier"},
                "display_name": {"type": "string", "description": "Human-readable name"},
                "generation": {"type": "integer", "description": "-1=origin, 0=primary, 1+=descendant"},
                "lineage": {"type": "string", "description": "Lineage or group name"},
                "specialization": {"type": "string", "description": "What this agent specializes in"},
                "origin_story": {"type": "string", "description": "Agent description"},
                "color": {"type": "string", "description": "Hex color (e.g., #FFD700)"},
                "symbol": {"type": "string", "description": "Unicode symbol"},
            },
            "required": ["agent_id", "display_name", "generation", "lineage", "specialization"],
        },
    },
    "list_agents": {
        "description": "List all registered agents in the memory system.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },

    # -----------------------------------------------------------------
    # Health & Stats
    # -----------------------------------------------------------------
    "memory_health": {
        "description": "Get a health report for the memory system: total memories, links, episodes, and breakdowns by type and layer.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "memory_graph_stats": {
        "description": "Get detailed statistics about the memory graph: node counts, link counts by type, and graph structure metrics.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "memory_neighbors": {
        "description": "Get memories directly linked to a given memory, sorted by link strength.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory to get neighbors of"},
                "max_results": {"type": "integer", "description": "Max neighbors (default: 10)"},
            },
            "required": ["memory_id"],
        },
    },
    "cortex_stats": {
        "description": "Get comprehensive system statistics as raw JSON.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },

    # -----------------------------------------------------------------
    # Dream Engine (offline maintenance)
    # -----------------------------------------------------------------
    "dream_run": {
        "description": "Run an offline memory maintenance cycle. Consolidates recent episodes, extracts patterns, prunes low-value memories, and discovers new connections. Uses LLM calls.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_llm_calls": {"type": "integer", "description": "Max LLM calls for this cycle (default: 20)"},
            },
            "required": [],
        },
    },
    "dream_status": {
        "description": "Get the status and last report from the memory maintenance cycle.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },

    # -----------------------------------------------------------------
    # Graph exploration & advanced tools
    # -----------------------------------------------------------------
    "find_path": {
        "description": "Find the shortest chain of links between two memories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Starting memory ID"},
                "target_id": {"type": "string", "description": "Target memory ID"},
            },
            "required": ["source_id", "target_id"],
        },
    },
    "common_neighbors": {
        "description": "Find memories that are linked to both memory A and memory B.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id_a": {"type": "string", "description": "First memory ID"},
                "id_b": {"type": "string", "description": "Second memory ID"},
            },
            "required": ["id_a", "id_b"],
        },
    },
    "create_schema": {
        "description": "Create a general pattern or principle derived from multiple memories. Useful for capturing recurring themes or lessons learned.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The pattern or principle to record"},
                "source_ids": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "IDs of the memories this pattern is derived from"},
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Tags for categorization"},
                "agent_id": {"type": "string", "description": "Agent creating this schema"},
            },
            "required": ["content", "source_ids"],
        },
    },
    "list_schemas": {
        "description": "List all stored patterns and principles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Filter by agent"},
            },
            "required": [],
        },
    },
    "find_matching_schemas": {
        "description": "Find patterns and principles matching given tags or concepts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Tags to match"},
                "concepts": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Concepts to match"}
            },
            "required": [],
        },
    },
    "get_schema_sources": {
        "description": "Get the original memories that a pattern or principle was derived from.",
        "input_schema": {
            "type": "object",
            "properties": {
                "schema_id": {"type": "string", "description": "Schema memory ID"},
            },
            "required": ["schema_id"],
        },
    },
    "store_procedure": {
        "description": "Store a workflow, strategy, or how-to guide. These are recalled when you need instructions for a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The workflow or how-to content"},
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Tags for categorization"},
                "derived_from": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "IDs of memories this procedure is derived from"},
                "agent_id": {"type": "string", "description": "Agent storing this procedure"},
            },
            "required": ["content"],
        },
    },
    "list_procedures": {
        "description": "List all stored workflows, strategies, and how-to guides.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Filter by agent"},
                "min_salience": {"type": "number", "description": "Minimum importance threshold (default: 0.0)"},
            },
            "required": [],
        },
    },
    "find_relevant_procedures": {
        "description": "Find workflows and how-to guides matching given tags or concepts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Tags to match"},
                "concepts": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Concepts to match"}
            },
            "required": [],
        },
    },
    "record_procedure_outcome": {
        "description": "Record whether a procedure worked or failed. This improves future procedure recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "procedure_id": {"type": "string", "description": "Procedure memory ID"},
                "success": {"type": "boolean", "description": "Whether the procedure succeeded"},
            },
            "required": ["procedure_id", "success"],
        },
    },
    "emotional_summary": {
        "description": "Get a breakdown of memories by emotional tone (positive, negative, neutral, mixed).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },

    # -----------------------------------------------------------------
    # File ingestion
    # -----------------------------------------------------------------
    "ingest_file": {
        "description": "Read a file and store its contents as searchable memories. Supports text, markdown, JSON, images (PNG/JPG/WebP), PDFs, HTML, and CSV. Large files are automatically split into sections. Images get captions, OCR text, and optional vision embeddings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Tags to apply to all imported memories"},
                "agent_id": {"type": "string", "description": "Agent performing the import"},
                "session_id": {"type": "string", "description": "Session ID for episode tracking"},
            },
            "required": ["file_path"],
        },
    },
    "describe_image": {
        "description": "Generate a text description of an image file without storing it in memory. Uses local vision model (Ollama) or filename fallback.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Absolute path to the image file"},
            },
            "required": ["image_path"],
        },
    },
    "search_vision": {
        "description": "Search image memories by text description or find similar images. Requires vision extras installed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string", "description": "Text query to search image memories"},
                "image_path": {"type": "string", "description": "Path to an image to use as query (image-to-image search)"},
                "top_k": {"type": "integer", "description": "Max results (default: 10)"},
            },
        },
    },

    # -----------------------------------------------------------------
    # CRUD: Trash can
    # -----------------------------------------------------------------
    "list_deleted": {
        "description": "List soft-deleted memories that can be restored.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Filter by agent"},
                "limit": {"type": "integer", "description": "Max results (default: 50)"},
            },
            "required": [],
        },
    },
    "restore_memory": {
        "description": "Undelete a soft-deleted memory and re-add it to the vector index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to restore"},
                "agent_id": {"type": "string", "description": "Agent ID for access check"},
            },
            "required": ["memory_id"],
        },
    },
    "purge_memory": {
        "description": "Permanently delete a memory, bypassing the trash can.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to permanently delete"},
                "agent_id": {"type": "string", "description": "Agent ID for access check"},
            },
            "required": ["memory_id"],
        },
    },
    "purge_all_deleted": {
        "description": "Hard-delete all soft-deleted memories older than N days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Only purge this agent's deleted memories"},
                "older_than_days": {"type": "integer", "description": "Age threshold in days (default: 30)"},
            },
            "required": [],
        },
    },

    # -----------------------------------------------------------------
    # CRUD: Versions
    # -----------------------------------------------------------------
    "get_memory_versions": {
        "description": "Get version history for a memory. Each content change creates a snapshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID"},
                "limit": {"type": "integer", "description": "Max versions to return (default: 10)"},
            },
            "required": ["memory_id"],
        },
    },
    "restore_version": {
        "description": "Restore a memory to a previous version snapshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "version_id": {"type": "integer", "description": "Version row ID from get_memory_versions"},
                "agent_id": {"type": "string", "description": "Agent ID for access check"},
            },
            "required": ["version_id"],
        },
    },

    # -----------------------------------------------------------------
    # Tag management
    # -----------------------------------------------------------------
    "list_tags": {
        "description": "List all tags with usage counts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Filter by agent"},
            },
            "required": [],
        },
    },
    "rename_tag": {
        "description": "Rename a tag across all memories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "old_tag": {"type": "string", "description": "Current tag name"},
                "new_tag": {"type": "string", "description": "New tag name"},
                "agent_id": {"type": "string", "description": "Only rename for this agent"},
            },
            "required": ["old_tag", "new_tag"],
        },
    },
    "merge_tags": {
        "description": "Merge multiple tags into one.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_tags": {"type": "array", "items": {"type": "string"}, "description": "Tags to merge"},
                "target_tag": {"type": "string", "description": "Destination tag"},
                "agent_id": {"type": "string", "description": "Only merge for this agent"},
            },
            "required": ["source_tags", "target_tag"],
        },
    },
    "delete_tag": {
        "description": "Remove a tag from all memories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tag": {"type": "string", "description": "Tag to remove"},
                "agent_id": {"type": "string", "description": "Only remove for this agent"},
            },
            "required": ["tag"],
        },
    },

    # -----------------------------------------------------------------
    # Bulk operations
    # -----------------------------------------------------------------
    "bulk_delete": {
        "description": "Delete multiple memories at once. Defaults to soft delete.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_ids": {"type": "array", "items": {"type": "string"}, "description": "Memory IDs to delete"},
                "hard": {"type": "boolean", "description": "If true, permanently delete"},
                "agent_id": {"type": "string", "description": "Agent ID for access check"},
            },
            "required": ["memory_ids"],
        },
    },
    "export_memories": {
        "description": "Export memories to JSON or Markdown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_ids": {"type": "array", "items": {"type": "string"}, "description": "Specific memory IDs (omit for all visible)"},
                "agent_id": {"type": "string", "description": "Agent ID for scope"},
                "format": {"type": "string", "enum": ["json", "markdown"], "description": "Export format"},
            },
            "required": [],
        },
    },

    # -----------------------------------------------------------------
    # Thread management
    # -----------------------------------------------------------------
    "list_threads": {
        "description": "List conversation threads with memory counts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Filter by agent"},
                "limit": {"type": "integer", "description": "Max threads (default: 50)"},
            },
            "required": [],
        },
    },
    "get_thread_memories": {
        "description": "Get all memories in a conversation thread.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thread_id": {"type": "string", "description": "Conversation thread ID"},
                "limit": {"type": "integer", "description": "Max memories (default: 100)"},
            },
            "required": ["thread_id"],
        },
    },
    "prune_thread": {
        "description": "Soft-delete all memories in a conversation thread.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thread_id": {"type": "string", "description": "Thread ID to prune"},
                "agent_id": {"type": "string", "description": "Agent ID for access check"},
            },
            "required": ["thread_id"],
        },
    },

    # -----------------------------------------------------------------
    # Agent Messaging
    # -----------------------------------------------------------------
    "send_message": {
        "description": "Send a message to another agent. Bypasses gating — messages are always delivered. Auto-tags with from/to and links replies. Use to='all' for broadcast.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient agent ID (e.g. 'CLAUDE-HAILO'), or 'all' for broadcast"},
                "content": {"type": "string", "description": "Message content"},
                "in_reply_to": {"type": "string", "description": "Memory ID of message being replied to (creates a supports link)"},
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}], "description": "Additional tags"}
            },
            "required": ["to", "content"],
        },
    },
    "check_inbox": {
        "description": "Check for messages from other agents addressed to you. Returns newest first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_agent": {"type": "string", "description": "Only show messages from this agent"},
                "limit": {"type": "integer", "description": "Max messages to return (default: 10)"},
                "since": {"type": "string", "description": "Only messages after this ISO timestamp (e.g. '2026-02-08T00:00:00')"},
            },
        },
    },

    # -----------------------------------------------------------------
    # Aliases (backward-compatible)
    # -----------------------------------------------------------------
    "memory_store": {
        "description": "Save information to memory (alias for 'remember').",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content"},
                "visibility": {"type": "string", "enum": ["private", "shared", "thread"]},
                "message_type": {"type": "string", "description": "Type of message"},
                "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]},
                "conversation_thread": {"type": "string"},
                "related_agents": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]},
                "responding_to": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]}
            },
            "required": ["content"],
        },
    },
    "memory_search": {
        "description": "Search memories (alias for 'recall').",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "agent_filter": {"type": "string"},
                "visibility": {"type": "string", "enum": ["shared", "private", "all"]},
                "n_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    },

    # -----------------------------------------------------------------
    # Cognitive Bootstrap (CCBS)
    # -----------------------------------------------------------------
    "cognitive_bootstrap": {
        "description": "Assemble a dynamic cognitive prompt block from Cerebro Cognitive Bootstrap System (CCBS) modules. Analyzes your query to load relevant cognitive priming modules (soul manifest, core identity, technical, analysis, creative, etc.) within a token budget.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Your intent or question — drives module auto-selection"},
                "mode": {"type": "string", "enum": ["minimal", "standard", "full"], "description": "Token budget tier: minimal (~900), standard (~1600), full (~4200). Auto-detected if omitted."},
                "max_tokens": {"type": "integer", "description": "Hard token ceiling (default: 4000)"},
                "agent_id": {"type": "string", "description": "Agent ID for module visibility filtering"},
            },
            "required": ["query"],
        },
    },

    # -----------------------------------------------------------------
    # Near-duplicate detection
    # -----------------------------------------------------------------
    "check_near_duplicates": {
        "description": "Check if content would be a near-duplicate of existing memories. Returns existing memories with similarity scores above the threshold. Useful for deduplication previews before ingestion.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to check for near-duplicates"},
                "threshold": {"type": "number", "description": "Minimum cosine similarity (0.0-1.0). Higher = stricter. Default: 0.95"},
                "agent_id": {"type": "string", "description": "Filter by agent visibility"},
                "top_k": {"type": "integer", "description": "Max results per collection. Default: 5"},
            },
            "required": ["content"],
        },
    },

    # -----------------------------------------------------------------
    # Activation & Decay visualization
    # -----------------------------------------------------------------
    "activation_heatmap": {
        "description": "Get activation data for all memories (age, activation, retrievability, layer, salience). Suitable for scatter-plot visualization of memory health.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max memories to analyze. Default: 200"},
                "agent_id": {"type": "string", "description": "Filter by agent visibility"},
            },
            "required": [],
        },
    },
    "activation_at_risk": {
        "description": "Get memories that are fading — low activation, low retrievability, or not accessed recently. These memories may need review or revival.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {"type": "integer", "description": "Hours since last access to be considered 'at risk'. Default: 24"},
                "layer": {"type": "string", "enum": ["sensory", "working", "long_term"], "description": "Filter by memory layer"},
                "agent_id": {"type": "string", "description": "Filter by agent visibility"},
                "limit": {"type": "integer", "description": "Max results. Default: 50"},
            },
            "required": [],
        },
    },
    "activation_curve": {
        "description": "Get the projected ACT-R decay curve for a specific memory over N days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to analyze"},
                "days": {"type": "integer", "description": "Days to project. Default: 30"},
            },
            "required": ["memory_id"],
        },
    },

    # -----------------------------------------------------------------
    # Audit logging
    # -----------------------------------------------------------------
    "query_audit": {
        "description": "Query the audit log with optional filters (event type, actor, target). Returns timestamped records of memory mutations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_type": {"type": "string", "description": "Filter by event type (e.g. memory_deleted, content_edited, visibility_changed)"},
                "actor": {"type": "string", "description": "Filter by agent ID who performed the action"},
                "target": {"type": "string", "description": "Filter by target memory ID"},
                "limit": {"type": "integer", "description": "Max entries. Default: 50"},
                "offset": {"type": "integer", "description": "Pagination offset. Default: 0"},
            },
            "required": [],
        },
    },
    "audit_summary": {
        "description": "Get a summary of audit events grouped by type with counts.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


# =============================================================================
# Tool handlers
# =============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    tools = []
    for name, schema in TOOL_SCHEMAS.items():
        tools.append(Tool(
            name=name,
            description=schema["description"],
            inputSchema=schema.get("input_schema", {
                "type": "object", "properties": {}, "required": [],
            }),
        ))
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    logger.info(f"Tool call: {name} args={list(arguments.keys())}")
    cortex = get_cortex()

    try:
        # =============================================================
        # Core: remember / recall / associate
        # =============================================================
        if name == "remember" or name == "memory_store":
            return await _handle_remember(cortex, arguments)

        elif name == "recall" or name == "memory_search":
            return await _handle_recall(cortex, arguments)

        elif name == "associate":
            return await _handle_associate(cortex, arguments)

        # =============================================================
        # Memory CRUD
        # =============================================================
        elif name == "get_memory":
            return await _handle_get_memory(cortex, arguments)

        elif name == "delete_memory":
            return await _handle_delete_memory(cortex, arguments)

        elif name == "update_memory":
            return await _handle_update_memory(cortex, arguments)

        elif name == "share_memory":
            return await _handle_share_memory(cortex, arguments)

        # =============================================================
        # Episodes
        # =============================================================
        elif name == "episode_start":
            return await _handle_episode_start(cortex, arguments)

        elif name == "episode_add_step":
            return await _handle_episode_add_step(cortex, arguments)

        elif name == "episode_end":
            return await _handle_episode_end(cortex, arguments)

        # =============================================================
        # Episodes & Intentions (Phase B)
        # =============================================================
        elif name == "list_episodes":
            return await _handle_list_episodes(cortex, arguments)

        elif name == "get_episode":
            return await _handle_get_episode(cortex, arguments)

        elif name == "get_episode_memories":
            return await _handle_get_episode_memories(cortex, arguments)

        elif name == "store_intention":
            return await _handle_store_intention(cortex, arguments)

        elif name == "list_intentions":
            return await _handle_list_intentions(cortex, arguments)

        elif name == "resolve_intention":
            return await _handle_resolve_intention(cortex, arguments)

        # =============================================================
        # Sessions
        # =============================================================
        elif name == "session_save":
            return await _handle_session_save(cortex, arguments)

        elif name == "session_recall":
            return await _handle_session_recall(cortex, arguments)

        # =============================================================
        # Agent Messaging
        # =============================================================
        elif name == "send_message":
            return await _handle_send_message(cortex, arguments)

        elif name == "check_inbox":
            return await _handle_check_inbox(cortex, arguments)

        # =============================================================
        # Agents
        # =============================================================
        elif name == "register_agent":
            return await _handle_register_agent(cortex, arguments)

        elif name == "list_agents":
            return await _handle_list_agents(cortex)

        # =============================================================
        # Health & Stats
        # =============================================================
        elif name == "memory_health":
            return await _handle_memory_health(cortex)

        elif name == "memory_graph_stats":
            return await _handle_graph_stats(cortex)

        elif name == "memory_neighbors":
            return await _handle_neighbors(cortex, arguments)

        elif name == "cortex_stats":
            return await _handle_cortex_stats(cortex)

        # =============================================================
        # Dream Engine
        # =============================================================
        elif name == "dream_run":
            return await _handle_dream_run(cortex, arguments)

        elif name == "dream_status":
            return await _handle_dream_status(cortex)

        # =============================================================
        # Engine Capabilities (Phase D)
        # =============================================================
        elif name == "find_path":
            return await _handle_find_path(cortex, arguments)

        elif name == "common_neighbors":
            return await _handle_common_neighbors(cortex, arguments)

        elif name == "create_schema":
            return await _handle_create_schema(cortex, arguments)

        elif name == "list_schemas":
            return await _handle_list_schemas(cortex, arguments)

        elif name == "find_matching_schemas":
            return await _handle_find_matching_schemas(cortex, arguments)

        elif name == "get_schema_sources":
            return await _handle_get_schema_sources(cortex, arguments)

        elif name == "store_procedure":
            return await _handle_store_procedure(cortex, arguments)

        elif name == "list_procedures":
            return await _handle_list_procedures(cortex, arguments)

        elif name == "find_relevant_procedures":
            return await _handle_find_relevant_procedures(cortex, arguments)

        elif name == "record_procedure_outcome":
            return await _handle_record_procedure_outcome(cortex, arguments)

        elif name == "emotional_summary":
            return await _handle_emotional_summary(cortex, arguments)

        elif name == "ingest_file":
            return await _handle_ingest_file(cortex, arguments)

        elif name == "describe_image":
            return await _handle_describe_image(cortex, arguments)

        elif name == "search_vision":
            return await _handle_search_vision(cortex, arguments)

        elif name == "list_deleted":
            return await _handle_list_deleted(cortex, arguments)

        elif name == "restore_memory":
            return await _handle_restore_memory(cortex, arguments)

        elif name == "purge_memory":
            return await _handle_purge_memory(cortex, arguments)

        elif name == "purge_all_deleted":
            return await _handle_purge_all_deleted(cortex, arguments)

        elif name == "get_memory_versions":
            return await _handle_get_memory_versions(cortex, arguments)

        elif name == "restore_version":
            return await _handle_restore_version(cortex, arguments)

        elif name == "list_tags":
            return await _handle_list_tags(cortex, arguments)

        elif name == "rename_tag":
            return await _handle_rename_tag(cortex, arguments)

        elif name == "merge_tags":
            return await _handle_merge_tags(cortex, arguments)

        elif name == "delete_tag":
            return await _handle_delete_tag(cortex, arguments)

        elif name == "bulk_delete":
            return await _handle_bulk_delete(cortex, arguments)

        elif name == "export_memories":
            return await _handle_export_memories(cortex, arguments)

        elif name == "list_threads":
            return await _handle_list_threads(cortex, arguments)

        elif name == "get_thread_memories":
            return await _handle_get_thread_memories(cortex, arguments)

        elif name == "prune_thread":
            return await _handle_prune_thread(cortex, arguments)

        # =============================================================
        # Cognitive Bootstrap (CCBS)
        # =============================================================
        elif name == "cognitive_bootstrap":
            return await _handle_cognitive_bootstrap(cortex, arguments)

        # =============================================================
        # Near-duplicate detection
        # =============================================================
        elif name == "check_near_duplicates":
            return await _handle_check_near_duplicates(cortex, arguments)

        # =============================================================
        # Activation & Decay
        # =============================================================
        elif name == "activation_heatmap":
            return await _handle_activation_heatmap(cortex, arguments)

        elif name == "activation_at_risk":
            return await _handle_activation_at_risk(cortex, arguments)

        elif name == "activation_curve":
            return await _handle_activation_curve(cortex, arguments)

        # =============================================================
        # Audit logging
        # =============================================================
        elif name == "query_audit":
            return await _handle_query_audit(cortex, arguments)

        elif name == "audit_summary":
            return await _handle_audit_summary(cortex, arguments)

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool error [{name}]: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# =============================================================================
# Handler implementations
# =============================================================================

async def _handle_cognitive_bootstrap(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    from cerebro.interfaces.api_server import CognitiveBootstrapAssembler

    assembler = CognitiveBootstrapAssembler(cortex)
    result = assembler.assemble(
        query=args["query"],
        mode=args.get("mode"),
        agent_id=args.get("agent_id"),
        max_tokens=args.get("max_tokens", 4000),
    )

    lines = [
        f"**Cognitive Bootstrap** — mode: `{result['mode']}` | trigger: `{result.get('trigger') or 'auto'}` | tokens: {result['total_tokens']}/{result['max_tokens']}",
        f"Modules loaded ({len(result['modules_loaded'])}): {', '.join(result['modules_loaded'])}",
    ]
    if result['modules_missing']:
        lines.append(f"Modules missing: {', '.join(result['modules_missing'])}")

    lines.append(f"\n{result['assembled_block']}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_check_near_duplicates(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    matches = cortex.find_near_duplicates(
        content=args["content"],
        threshold=args.get("threshold", 0.95),
        agent_id=args.get("agent_id"),
        top_k=args.get("top_k", 5),
    )

    if not matches:
        return [TextContent(type="text", text="No near-duplicates found.")]

    lines = [f"**{len(matches)} near-duplicate(s) found:**\n"]
    for i, m in enumerate(matches, 1):
        sim = m.get("similarity", 0.0)
        preview = m.get("content", "")[:120]
        coll = m.get("collection", "unknown")
        lines.append(f"{i}. [{m.get('id', '?')}] similarity={sim:.3f} coll={coll} | {preview}...")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_activation_heatmap(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    import math
    from cerebro.activation.strength import base_level_activation, retrievability

    limit = args.get("limit", 200)
    agent_id = args.get("agent_id")

    rows = cortex._graph.conn.execute(
        "SELECT id FROM memory_nodes WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()

    now = time.time()
    points = []
    for row in rows:
        node = cortex._graph.get_node(row["id"])
        if not node:
            continue
        if not cortex._can_access(node, agent_id):
            continue

        strength = node.strength
        last_access = strength.access_timestamps[-1] if strength.access_timestamps else (
            node.created_at.timestamp() if hasattr(node.created_at, "timestamp") else now
        )
        age_hours = (now - last_access) / 3600.0
        activation = base_level_activation(
            strength.access_timestamps, now,
            strength.compressed_count, strength.compressed_avg_interval,
        )
        if math.isinf(activation):
            activation = -10.0
        r = retrievability(age_hours / 24.0, strength.stability)

        points.append({
            "id": node.id,
            "age_hours": round(age_hours, 1),
            "activation": round(activation, 3),
            "retrievability": round(r, 3),
            "layer": node.metadata.layer.value,
            "salience": node.metadata.salience,
            "access_count": strength.access_count,
        })

    lines = [f"**Activation Heatmap** — {len(points)} memories analyzed\n"]
    for p in points[:20]:
        lines.append(
            f"  [{p['id'][:8]}] age={p['age_hours']:.1f}h act={p['activation']:.2f} "
            f"ret={p['retrievability']:.2f} layer={p['layer']} salience={p['salience']:.2f}"
        )
    if len(points) > 20:
        lines.append(f"  ... and {len(points) - 20} more")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_activation_at_risk(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    at_risk = cortex.get_at_risk_memories(
        hours=args.get("hours", 24),
        layer=args.get("layer"),
        agent_id=args.get("agent_id"),
        limit=args.get("limit", 50),
    )

    if not at_risk:
        return [TextContent(type="text", text="No at-risk memories found.")]

    lines = [f"**{len(at_risk)} at-risk memory/memories:**\n"]
    for i, (node, activation, r, hours_since) in enumerate(at_risk, 1):
        preview = node.content[:120] + "..." if len(node.content) > 120 else node.content
        lines.append(
            f"{i}. [{node.id[:8]}] act={activation:.2f} ret={r:.2f} "
            f"{hours_since:.1f}h since access | {preview}"
        )
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_activation_curve(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    from cerebro.activation.strength import compute_activation_curve

    node = cortex.get_memory(args["memory_id"])
    if not node:
        return [TextContent(type="text", text=f"Memory not found: {args['memory_id']}")]

    days = args.get("days", 30)
    curve = compute_activation_curve(node.strength, days=days)

    lines = [
        f"**Activation Curve** for {args['memory_id']} — {days} days projection",
        f"Current access count: {node.strength.access_count}",
        f"Stability: {node.strength.stability:.2f}",
        "",
    ]
    for point in curve[:10]:
        lines.append(f"  Day {point['day']:>3}: activation={point['activation']:.3f}  retrievability={point['retrievability']:.3f}")
    if len(curve) > 10:
        lines.append(f"  ... and {len(curve) - 10} more days")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_query_audit(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    entries = cortex._graph.query_audit(
        event_type=args.get("event_type"),
        actor=args.get("actor"),
        target=args.get("target"),
        limit=args.get("limit", 50),
        offset=args.get("offset", 0),
    )
    total = cortex._graph.count_audit(
        event_type=args.get("event_type"),
        actor=args.get("actor"),
    )

    if not entries:
        return [TextContent(type="text", text="No audit entries found.")]

    lines = [f"**Audit Log** — {len(entries)} of {total} entries\n"]
    for e in entries:
        ts = e.get("timestamp", "unknown")
        et = e.get("event_type", "unknown")
        actor = e.get("actor_agent_id") or "system"
        target = e.get("target_memory_id") or "-"
        lines.append(f"  [{ts}] {et:20} | actor={actor:12} | target={target}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_audit_summary(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    rows = cortex._graph.conn.execute(
        "SELECT event_type, COUNT(*) as c FROM audit_log GROUP BY event_type ORDER BY c DESC"
    ).fetchall()

    if not rows:
        return [TextContent(type="text", text="No audit events recorded.")]

    total = sum(r["c"] for r in rows)
    lines = [f"**Audit Summary** — {total} total events\n"]
    for r in rows:
        lines.append(f"  {r['event_type']}: {r['c']}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_remember(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    # Map Neo-Cortex args to CerebroCortex args
    memory_type = None
    if "memory_type" in args:
        try:
            memory_type = MemoryType(args["memory_type"])
        except ValueError:
            pass
    elif "message_type" in args:
        # Neo-Cortex compat: map message_type to memory_type
        type_map = {
            "fact": MemoryType.SEMANTIC,
            "dialogue": MemoryType.EPISODIC,
            "observation": MemoryType.SEMANTIC,
            "question": MemoryType.PROSPECTIVE,
            "cultural": MemoryType.SCHEMATIC,
            "discovery": MemoryType.SEMANTIC,
        }
        memory_type = type_map.get(args["message_type"])

    visibility = Visibility.SHARED
    if "visibility" in args:
        try:
            visibility = Visibility(args["visibility"])
        except ValueError:
            pass

    node = cortex.remember(
        content=args["content"],
        memory_type=memory_type,
        tags=_coerce_array(args.get("tags")),
        salience=args.get("salience"),
        agent_id=args.get("agent_id", DEFAULT_AGENT_ID),
        session_id=args.get("session_id"),
        visibility=visibility,
        context_ids=_coerce_array(args.get("context_ids") or args.get("responding_to")),
    )

    if node is None:
        return [TextContent(type="text", text="Memory gated out (duplicate or too short).")]

    return [TextContent(
        type="text",
        text=(
            f"Stored memory (ID: {node.id})\n"
            f"Type: {node.metadata.memory_type.value} | "
            f"Layer: {node.metadata.layer.value} | "
            f"Salience: {node.metadata.salience:.2f} | "
            f"Valence: {node.metadata.valence.value if hasattr(node.metadata.valence, 'value') else node.metadata.valence}\n"
            f"Concepts: {', '.join(node.metadata.concepts[:5]) if node.metadata.concepts else 'none'}\n"
            f"Links created: {node.link_count}"
        ),
    )]


async def _handle_recall(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    memory_types = None
    if "memory_types" in args:
        memory_types = [MemoryType(t) for t in _coerce_array(args["memory_types"]) or []]

    explain = args.get("explain", False)

    results = cortex.recall(
        query=args["query"],
        top_k=args.get("top_k", args.get("n_results", 10)),
        memory_types=memory_types,
        agent_id=args.get("agent_id", args.get("agent_filter")),
        min_salience=args.get("min_salience", 0.0),
        context_ids=_coerce_array(args.get("context_ids")),
        conversation_thread=args.get("conversation_thread"),
        explain=explain,
    )

    if not results:
        return [TextContent(type="text", text="No memories found.")]

    lines = [f"**Found {len(results)} memories:**\n"]
    for i, entry in enumerate(results, 1):
        node, score = entry[0], entry[1]
        explanation = entry[2] if explain and len(entry) > 2 else None

        content_preview = node.content[:150] + "..." if len(node.content) > 150 else node.content
        lines.append(
            f"{i}. [{node.metadata.memory_type.value}] (score: {score:.3f}, "
            f"salience: {node.metadata.salience:.2f}) "
            f"{content_preview}"
        )
        if explanation:
            lines.append(
                f"   | vector={explanation['vector_similarity']:.3f} "
                f"activation={explanation['actr_activation_score']:.3f} "
                f"retrievability={explanation['fsrs_retrievability']:.3f} "
                f"salience={explanation['salience']:.3f} "
                f"| layer={explanation['layer']} "
                f"accesses={explanation['access_count']} "
                f"age={explanation['age_hours']:.0f}h "
                f"stability={explanation['fsrs_stability_days']}d"
            )

    # Check for contradictions among results
    result_ids = [entry[0].id for entry in results]
    contradictions = cortex.find_contradictions_in_set(result_ids)
    if contradictions:
        lines.append("\n**Contradictions detected:**")
        seen_pairs: set[tuple[str, str]] = set()
        for mid, contra_ids in contradictions.items():
            for cid in contra_ids:
                pair = tuple(sorted([mid, cid]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    lines.append(f"  - {pair[0][:12]}... contradicts {pair[1][:12]}...")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_associate(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    link_id = cortex.associate(
        source_id=args["source_id"],
        target_id=args["target_id"],
        link_type=LinkType(args["link_type"]),
        weight=args.get("weight", 0.5),
        evidence=args.get("evidence"),
    )
    if link_id is None:
        return [TextContent(type="text", text="Failed: one or both memory IDs not found.")]

    return [TextContent(
        type="text",
        text=f"Link created (ID: {link_id}) {args['source_id']} --[{args['link_type']}]--> {args['target_id']}",
    )]


async def _handle_get_memory(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    node = cortex.get_memory(args["memory_id"], agent_id=args.get("agent_id"))
    if not node:
        return [TextContent(type="text", text=f"Memory not found: {args['memory_id']}")]
    return [TextContent(type="text", text=json.dumps({
        "id": node.id,
        "content": node.content,
        "type": node.metadata.memory_type.value,
        "layer": node.metadata.layer.value,
        "salience": round(node.metadata.salience, 3),
        "valence": node.metadata.valence.value if hasattr(node.metadata.valence, "value") else str(node.metadata.valence),
        "arousal": round(node.metadata.arousal, 3),
        "tags": node.metadata.tags,
        "concepts": node.metadata.concepts,
        "agent_id": node.metadata.agent_id,
        "visibility": node.metadata.visibility.value if hasattr(node.metadata.visibility, "value") else str(node.metadata.visibility),
        "created_at": node.created_at.isoformat(),
        "access_count": node.strength.access_count,
    }, indent=2))]


async def _handle_delete_memory(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    success = cortex.delete_memory(args["memory_id"], agent_id=args.get("agent_id"))
    if not success:
        return [TextContent(type="text", text=f"Memory not found: {args['memory_id']}")]
    return [TextContent(type="text", text=f"Deleted memory: {args['memory_id']}")]


async def _handle_update_memory(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    visibility = None
    if "visibility" in args:
        try:
            visibility = Visibility(args["visibility"])
        except ValueError:
            pass

    updated = cortex.update_memory(
        memory_id=args["memory_id"],
        content=args.get("content"),
        tags=_coerce_array(args.get("tags")),
        salience=args.get("salience"),
        visibility=visibility,
        agent_id=args.get("agent_id"),
    )
    if updated is None:
        return [TextContent(type="text", text=f"Memory not found: {args['memory_id']}")]
    return [TextContent(type="text", text=(
        f"Updated memory: {updated.id}\n"
        f"Type: {updated.metadata.memory_type.value} | "
        f"Salience: {updated.metadata.salience:.2f} | "
        f"Tags: {', '.join(updated.metadata.tags)}"
    ))]


async def _handle_share_memory(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    try:
        new_vis = Visibility(args["visibility"])
    except ValueError:
        return [TextContent(type="text", text=f"Invalid visibility: {args['visibility']}")]

    updated = cortex.share_memory(
        memory_id=args["memory_id"],
        new_visibility=new_vis,
        agent_id=args.get("agent_id"),
    )
    if updated is None:
        return [TextContent(type="text", text=f"Not found or not authorized: {args['memory_id']}")]
    return [TextContent(
        type="text",
        text=f"Visibility changed: {args['memory_id']} -> {new_vis.value}",
    )]


async def _handle_episode_start(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    episode = cortex.episode_start(
        title=args.get("title"),
        session_id=args.get("session_id"),
        agent_id=args.get("agent_id", DEFAULT_AGENT_ID),
    )
    return [TextContent(
        type="text",
        text=f"Episode started (ID: {episode.id})",
    )]


async def _handle_episode_add_step(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    step = cortex.episodes.add_step(
        episode_id=args["episode_id"],
        memory_id=args["memory_id"],
        role=args.get("role", "event"),
    )
    if step is None:
        return [TextContent(type="text", text="Failed: episode not found.")]

    return [TextContent(
        type="text",
        text=f"Step added at position {step.position} (role: {step.role})",
    )]


async def _handle_episode_end(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    valence = EmotionalValence.NEUTRAL
    if "valence" in args:
        try:
            valence = EmotionalValence(args["valence"])
        except ValueError:
            pass

    episode = cortex.episode_end(
        episode_id=args["episode_id"],
        summary=args.get("summary"),
        valence=valence,
    )
    if episode is None:
        return [TextContent(type="text", text="Failed: episode not found.")]

    return [TextContent(
        type="text",
        text=f"Episode ended (ID: {episode.id}, steps: {len(episode.steps)})",
    )]


async def _handle_list_episodes(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    limit = args.get("limit", 10)
    agent_id = args.get("agent_id")
    episodes = cortex.list_episodes(limit=limit, agent_id=agent_id)
    if not episodes:
        return [TextContent(type="text", text="No episodes found.")]

    lines = [f"**{len(episodes)} Episodes:**\n"]
    for ep in episodes:
        title = ep.title or "(untitled)"
        valence = ep.overall_valence.value if hasattr(ep.overall_valence, "value") else str(ep.overall_valence)
        created = ep.created_at.strftime("%Y-%m-%d %H:%M") if ep.created_at else "?"
        lines.append(
            f"- {ep.id}: {title} | steps: {len(ep.steps)} | "
            f"valence: {valence} | created: {created}"
        )
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_get_episode(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    episode = cortex.get_episode(args["episode_id"], agent_id=args.get("agent_id"))
    if episode is None:
        return [TextContent(type="text", text=f"Episode not found: {args['episode_id']}")]

    data = {
        "id": episode.id,
        "title": episode.title,
        "steps": [
            {
                "memory_id": s.memory_id,
                "position": s.position,
                "role": s.role,
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
            }
            for s in episode.steps
        ],
        "session_id": episode.session_id,
        "agent_id": episode.agent_id,
        "tags": episode.tags,
        "started_at": episode.started_at.isoformat() if episode.started_at else None,
        "ended_at": episode.ended_at.isoformat() if episode.ended_at else None,
        "overall_valence": episode.overall_valence.value if hasattr(episode.overall_valence, "value") else str(episode.overall_valence),
        "consolidated": episode.consolidated,
        "created_at": episode.created_at.isoformat() if episode.created_at else None,
    }
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


async def _handle_get_episode_memories(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    memories = cortex.get_episode_memories(args["episode_id"], agent_id=args.get("agent_id"))
    if not memories:
        return [TextContent(type="text", text=f"No memories found for episode: {args['episode_id']}")]

    lines = [f"**{len(memories)} memories in episode {args['episode_id']}:**\n"]
    for i, node in enumerate(memories, 1):
        preview = node.content[:120] + "..." if len(node.content) > 120 else node.content
        lines.append(f"{i}. [{node.metadata.memory_type.value}] {node.id}: {preview}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_store_intention(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    node = cortex.store_intention(
        content=args["content"],
        tags=_coerce_array(args.get("tags")),
        agent_id=args.get("agent_id", DEFAULT_AGENT_ID),
        salience=args.get("salience", 0.7),
    )
    preview = node.content[:80] + "..." if len(node.content) > 80 else node.content
    return [TextContent(
        type="text",
        text=f"Stored intention (ID: {node.id}) {preview}",
    )]


async def _handle_list_intentions(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    agent_id = args.get("agent_id")
    min_salience = args.get("min_salience", 0.3)
    intentions = cortex.list_intentions(agent_id=agent_id, min_salience=min_salience)
    if not intentions:
        return [TextContent(type="text", text="No pending intentions.")]

    lines = [f"**{len(intentions)} Pending Intentions:**\n"]
    for i, node in enumerate(intentions, 1):
        preview = node.content[:120] + "..." if len(node.content) > 120 else node.content
        lines.append(f"{i}. (salience: {node.metadata.salience:.2f}) {preview}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_resolve_intention(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    success = cortex.resolve_intention(args["memory_id"])
    if not success:
        return [TextContent(type="text", text=f"Intention not found: {args['memory_id']}")]
    return [TextContent(type="text", text=f"Resolved intention: {args['memory_id']}")]


async def _handle_session_save(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    # Build session note content
    parts = [f"SESSION SUMMARY: {args['session_summary']}"]

    key_discoveries = _coerce_array(args.get("key_discoveries"))
    unfinished_business = _coerce_array(args.get("unfinished_business"))
    if_disoriented = _coerce_array(args.get("if_disoriented"))

    if key_discoveries:
        parts.append("\nKEY DISCOVERIES:")
        for d in key_discoveries:
            parts.append(f"  - {d}")

    if unfinished_business:
        parts.append("\nUNFINISHED BUSINESS:")
        for u in unfinished_business:
            parts.append(f"  - {u}")

    if if_disoriented:
        parts.append("\nIF DISORIENTED:")
        for o in if_disoriented:
            parts.append(f"  - {o}")

    content = "\n".join(parts)
    priority = args.get("priority", "MEDIUM")
    session_type = args.get("session_type", "orientation")

    node = cortex.remember(
        content=content,
        memory_type=MemoryType.EPISODIC,
        tags=[
            "session_note",
            f"priority:{priority}",
            f"session_type:{session_type}",
        ],
        salience={"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.4}.get(priority, 0.7),
    )

    if node is None:
        return [TextContent(type="text", text="Failed to save session note (duplicate?).")]

    # Auto-close stale episodes on session boundary
    closed = cortex.episodes.close_stale_episodes()
    extra = f", auto-closed {len(closed)} stale episodes" if closed else ""

    return [TextContent(
        type="text",
        text=f"Session note saved (ID: {node.id}, priority: {priority}{extra})",
    )]


async def _handle_session_recall(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    lookback_hours = args.get("lookback_hours", 168)
    limit = args.get("limit", 10)

    # Search for session notes by tag
    from datetime import timedelta
    since = datetime.now() - timedelta(hours=lookback_hours)

    nodes = cortex.graph.get_nodes_since(since)

    # Filter to session notes
    session_notes = []
    for node in nodes:
        if "session_note" not in node.metadata.tags:
            continue

        # Apply filters
        if args.get("priority_filter"):
            if f"priority:{args['priority_filter']}" not in node.metadata.tags:
                continue
        if args.get("session_type"):
            if f"session_type:{args['session_type']}" not in node.metadata.tags:
                continue

        session_notes.append(node)

    # Sort by creation time descending, limit
    session_notes.sort(key=lambda n: n.created_at, reverse=True)
    session_notes = session_notes[:limit]

    if not session_notes:
        return [TextContent(type="text", text="No session notes found.")]

    lines = [f"**{len(session_notes)} Session Notes:**\n"]
    for note in session_notes:
        # Extract priority from tags
        priority = "MEDIUM"
        for tag in note.metadata.tags:
            if tag.startswith("priority:"):
                priority = tag.split(":")[1]

        content_preview = note.content[:200]
        lines.append(
            f"---\n[{priority}] {note.created_at.strftime('%Y-%m-%d %H:%M')} (ID: {note.id})\n"
            f"{content_preview}{'...' if len(note.content) > 200 else ''}\n"
        )

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_register_agent(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    profile = AgentProfile(
        id=args["agent_id"],
        display_name=args["display_name"],
        generation=args["generation"],
        lineage=args["lineage"],
        specialization=args["specialization"],
        origin_story=args.get("origin_story"),
        color=args.get("color", "#888888"),
        symbol=args.get("symbol", "A"),
    )
    cortex.graph.register_agent(profile)
    return [TextContent(
        type="text",
        text=f"Agent registered: {profile.symbol} {profile.id} ({profile.display_name})",
    )]


async def _handle_list_agents(cortex: CerebroCortex) -> list[TextContent]:
    agents = cortex.graph.list_agents()
    if not agents:
        return [TextContent(type="text", text="No agents registered.")]

    lines = [f"**{len(agents)} Registered Agents:**\n"]
    for a in agents:
        lines.append(
            f"- {a.symbol} **{a.id}** ({a.display_name}) "
            f"- Gen {a.generation} - {a.specialization}"
        )
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_memory_health(cortex: CerebroCortex) -> list[TextContent]:
    stats = cortex.stats()

    # Run promotion sweep
    promotions = cortex.executive.run_promotion_sweep()

    # Get pending intentions
    intentions = cortex.executive.get_pending_intentions()

    lines = [
        "**CerebroCortex Health Report**\n",
        f"Memories: {stats['nodes']}",
        f"Links: {stats['links']}",
        f"Episodes: {stats['episodes']}",
        f"Schemas: {stats['schemas']}",
        f"\n**By Type:**",
    ]
    for mtype, count in stats.get("memory_types", {}).items():
        lines.append(f"  {mtype}: {count}")

    lines.append(f"\n**By Layer:**")
    for layer, count in stats.get("layers", {}).items():
        lines.append(f"  {layer}: {count}")

    if promotions:
        lines.append(f"\n**Promotions this check:**")
        for layer, count in promotions.items():
            lines.append(f"  -> {layer}: {count}")

    lines.append(f"\n**Pending intentions:** {len(intentions)}")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_graph_stats(cortex: CerebroCortex) -> list[TextContent]:
    stats = cortex.stats()

    lines = [
        "**Graph Statistics**\n",
        f"Nodes (SQLite): {stats['nodes']}",
        f"Links (SQLite): {stats['links']}",
        f"Vertices (igraph): {stats['igraph_vertices']}",
        f"Edges (igraph): {stats['igraph_edges']}",
        f"\n**Link Types:**",
    ]
    for ltype, count in stats.get("link_types", {}).items():
        lines.append(f"  {ltype}: {count}")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_neighbors(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    memory_id = args["memory_id"]
    max_results = args.get("max_results", 10)

    neighbors = cortex.links.get_strongest_connections(memory_id, top_n=max_results)
    if not neighbors:
        return [TextContent(type="text", text=f"No neighbors found for {memory_id}.")]

    lines = [f"**Neighbors of {memory_id}:**\n"]
    for neighbor_id, weight, link_type, *_ in neighbors:
        node = cortex.graph.get_node(neighbor_id)
        content_preview = node.content[:80] if node else "?"
        lines.append(f"  --[{link_type} w={weight:.2f}]--> {neighbor_id}: {content_preview}")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_cortex_stats(cortex: CerebroCortex) -> list[TextContent]:
    stats = cortex.stats()
    return [TextContent(
        type="text",
        text=json.dumps(stats, indent=2, default=str),
    )]


async def _handle_dream_run(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    dream = get_dream_engine(cortex)

    if dream.is_running:
        return [TextContent(type="text", text="Dream cycle already in progress.")]

    reports = dream.run_all_agents_cycle()
    lines = [f"**Dream Cycle Complete** ({len(reports)} agent(s))\n"]
    for report in reports:
        agent_label = report.agent_id or "unscoped"
        phase_summary = ", ".join(
            f"{p.phase.value}({'ok' if p.success else 'FAIL'})" for p in report.phases
        )
        lines.append(
            f"**{agent_label}**: {report.total_duration_seconds:.1f}s, "
            f"{report.episodes_consolidated} episodes, "
            f"{report.total_llm_calls} LLM calls, "
            f"success={report.success}\n  Phases: {phase_summary}"
        )

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_dream_status(cortex: CerebroCortex) -> list[TextContent]:
    dream = get_dream_engine(cortex)

    if dream.is_running:
        return [TextContent(type="text", text="Dream cycle in progress...")]

    report = dream.last_report
    if report is None:
        return [TextContent(type="text", text="No dream cycles have run yet.")]

    return [TextContent(
        type="text",
        text=json.dumps(report.to_dict(), indent=2),
    )]



# =============================================================================
# Engine Capabilities (Phase D) handlers
# =============================================================================

async def _handle_find_path(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    path = cortex.find_path(args["source_id"], args["target_id"])
    if not path:
        return [TextContent(type="text", text="No path found.")]
    return [TextContent(type="text", text=f"Path: {' -> '.join(path)}")]


async def _handle_common_neighbors(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    neighbor_ids = cortex.get_common_neighbors(args["id_a"], args["id_b"])
    if not neighbor_ids:
        return [TextContent(type="text", text="No common neighbors found.")]

    lines = [f"**{len(neighbor_ids)} Common Neighbors:**\n"]
    for nid in neighbor_ids:
        node = cortex.get_memory(nid)
        preview = node.content[:100] + "..." if node and len(node.content) > 100 else (node.content if node else "?")
        lines.append(f"- {nid}: {preview}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_create_schema(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    node = cortex.create_schema(
        content=args["content"],
        source_ids=_coerce_array(args["source_ids"]) or [],
        tags=_coerce_array(args.get("tags")),
        agent_id=args.get("agent_id", DEFAULT_AGENT_ID),
    )
    preview = node.content[:120] + "..." if len(node.content) > 120 else node.content
    return [TextContent(
        type="text",
        text=f"Created schema (ID: {node.id})\n{preview}",
    )]


async def _handle_list_schemas(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    schemas = cortex.list_schemas(agent_id=args.get("agent_id"))
    if not schemas:
        return [TextContent(type="text", text="No schemas found.")]

    lines = [f"**{len(schemas)} Schemas:**\n"]
    for i, node in enumerate(schemas, 1):
        preview = node.content[:120] + "..." if len(node.content) > 120 else node.content
        lines.append(f"{i}. {node.id}: {preview}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_find_matching_schemas(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    schemas = cortex.find_matching_schemas(
        tags=_coerce_array(args.get("tags")),
        concepts=_coerce_array(args.get("concepts")),
    )
    if not schemas:
        return [TextContent(type="text", text="No matching schemas found.")]

    lines = [f"**{len(schemas)} Matching Schemas:**\n"]
    for i, node in enumerate(schemas, 1):
        preview = node.content[:120] + "..." if len(node.content) > 120 else node.content
        lines.append(f"{i}. {node.id}: {preview}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_get_schema_sources(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    source_ids = cortex.get_schema_sources(args["schema_id"])
    if not source_ids:
        return [TextContent(type="text", text=f"No sources found for schema: {args['schema_id']}")]

    lines = [f"**{len(source_ids)} Source Memories for {args['schema_id']}:**\n"]
    for sid in source_ids:
        node = cortex.get_memory(sid)
        content = node.content[:120] + "..." if node and len(node.content) > 120 else (node.content if node else "?")
        lines.append(f"- {sid}: {content}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_store_procedure(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    node = cortex.store_procedure(
        content=args["content"],
        tags=_coerce_array(args.get("tags")),
        derived_from=_coerce_array(args.get("derived_from")),
        agent_id=args.get("agent_id", DEFAULT_AGENT_ID),
    )
    preview = node.content[:120] + "..." if len(node.content) > 120 else node.content
    return [TextContent(
        type="text",
        text=f"Stored procedure (ID: {node.id})\n{preview}",
    )]


async def _handle_list_procedures(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    procedures = cortex.list_procedures(
        agent_id=args.get("agent_id"),
        min_salience=args.get("min_salience", 0.0),
    )
    if not procedures:
        return [TextContent(type="text", text="No procedures found.")]

    lines = [f"**{len(procedures)} Procedures:**\n"]
    for i, node in enumerate(procedures, 1):
        preview = node.content[:120] + "..." if len(node.content) > 120 else node.content
        lines.append(f"{i}. (salience: {node.metadata.salience:.2f}) {node.id}: {preview}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_find_relevant_procedures(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    procedures = cortex.find_relevant_procedures(
        tags=_coerce_array(args.get("tags")),
        concepts=_coerce_array(args.get("concepts")),
    )
    if not procedures:
        return [TextContent(type="text", text="No matching procedures found.")]

    lines = [f"**{len(procedures)} Matching Procedures:**\n"]
    for i, node in enumerate(procedures, 1):
        preview = node.content[:120] + "..." if len(node.content) > 120 else node.content
        lines.append(f"{i}. (salience: {node.metadata.salience:.2f}) {node.id}: {preview}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_record_procedure_outcome(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    success = cortex.record_procedure_outcome(args["procedure_id"], args["success"])
    if not success:
        return [TextContent(type="text", text=f"Procedure not found: {args['procedure_id']}")]
    outcome = "success" if args["success"] else "failure"
    return [TextContent(
        type="text",
        text=f"Recorded outcome for {args['procedure_id']}: {outcome}",
    )]


async def _handle_emotional_summary(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    summary = cortex.get_emotional_summary()
    if not summary:
        return [TextContent(type="text", text="No emotional data available.")]

    lines = ["**Emotional Summary:**\n"]
    for valence, count in sorted(summary.items(), key=lambda x: -x[1]):
        lines.append(f"  {valence}: {count}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_ingest_file(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    file_path = Path(args["file_path"])
    if not file_path.exists():
        return [TextContent(type="text", text=f"File not found: {file_path}")]
    if not file_path.is_file():
        return [TextContent(type="text", text=f"Not a file: {file_path}")]

    from cerebro.ingestion import IngestionPipeline

    pipeline = IngestionPipeline(cortex)
    report = pipeline.ingest_file(
        file_path,
        tags=_coerce_array(args.get("tags")),
        agent_id=args.get("agent_id", DEFAULT_AGENT_ID),
        session_id=args.get("session_id"),
    )

    parts = [
        f"Ingested {file_path.name}: {report.memories_imported} memories stored",
    ]
    if report.attachments_created:
        parts.append(f"{len(report.attachments_created)} attachments created")
    if report.memories_skipped:
        parts.append(f"{report.memories_skipped} skipped")
    if report.errors:
        parts.append(f"{len(report.errors)} errors: {'; '.join(report.errors[:3])}")
    parts.append(f"({report.duration_seconds:.1f}s)")

    return [TextContent(type="text", text=" | ".join(parts))]


async def _handle_describe_image(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    from cerebro.ingestion.image_adapter import ImageAdapter

    path = Path(args["image_path"])
    if not path.exists():
        return [TextContent(type="text", text=f"Image not found: {path}")]

    adapter = ImageAdapter()
    caption = adapter._caption_image(path)
    ocr_text = adapter._ocr_image(path)

    lines = [f"Caption: {caption}"]
    if ocr_text:
        lines.append(f"OCR text: {ocr_text}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_search_vision(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    if cortex._vision_store is None:
        return [TextContent(
            type="text",
            text="Vision search not available. Install with: pip install cerebro-cortex[vision]"
        )]

    top_k = args.get("top_k", 10)
    results: list[dict] = []

    if args.get("image_path"):
        path = Path(args["image_path"])
        if not path.exists():
            return [TextContent(type="text", text=f"Image not found: {path}")]
        results = cortex._vision_store.search_by_image(path, n_results=top_k)
    elif args.get("query_text"):
        results = cortex._vision_store.search_by_text(args["query_text"], n_results=top_k)
    else:
        return [TextContent(type="text", text="Provide either query_text or image_path.")]

    if not results:
        return [TextContent(type="text", text="No vision matches found.")]

    lines = [f"**Found {len(results)} vision matches:**\n"]
    for i, hit in enumerate(results, 1):
        mem_id = hit.get("memory_id", "unknown")
        dist = hit.get("distance", 1.0)
        sim = max(0.0, 1.0 - dist)
        node = cortex.get_memory(mem_id) if mem_id != "unknown" else None
        preview = node.content[:120] + "..." if node and len(node.content) > 120 else (node.content if node else "")
        att_info = ""
        if node and node.metadata.attachments:
            att = node.metadata.attachments[0]
            att_info = f" | file: {att.file_path or 'unknown'}"
        lines.append(f"{i}. [{mem_id}] similarity={sim:.3f} {preview}{att_info}")

    return [TextContent(type="text", text="\n".join(lines))]


# =============================================================================
# CRUD handlers
# =============================================================================

async def _handle_list_deleted(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    nodes = cortex.list_deleted(
        agent_id=args.get("agent_id"),
        limit=args.get("limit", 50),
    )
    if not nodes:
        return [TextContent(type="text", text="No deleted memories found.")]
    lines = [f"**{len(nodes)} deleted memory(s):**\n"]
    for node in nodes:
        dt = node.metadata.deleted_at or "unknown"
        lines.append(f"- [{node.id}] deleted_at={dt} | {node.content[:100]}...")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_restore_memory(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    ok = cortex.restore_memory(args["memory_id"], agent_id=args.get("agent_id"))
    if ok:
        return [TextContent(type="text", text=f"Restored memory {args['memory_id']}.")]
    return [TextContent(type="text", text=f"Failed to restore {args['memory_id']}.")]


async def _handle_purge_memory(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    ok = cortex.purge_memory(args["memory_id"], agent_id=args.get("agent_id"))
    if ok:
        return [TextContent(type="text", text=f"Permanently deleted {args['memory_id']}.")]
    return [TextContent(type="text", text=f"Failed to purge {args['memory_id']}.")]


async def _handle_purge_all_deleted(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    count = cortex.purge_all_deleted(
        agent_id=args.get("agent_id"),
        older_than_days=args.get("older_than_days", 30),
    )
    return [TextContent(type="text", text=f"Purged {count} deleted memories.")]


async def _handle_get_memory_versions(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    versions = cortex.get_memory_versions(
        args["memory_id"],
        limit=args.get("limit", 10),
    )
    if not versions:
        return [TextContent(type="text", text="No versions found.")]
    lines = [f"**{len(versions)} version(s) for {args['memory_id']}:**\n"]
    for v in versions:
        lines.append(
            f"- v{v['id']} ({v['edited_at']}) salience={v['salience']:.2f} "
            f"visibility={v['visibility']} | {v['content'][:80]}..."
        )
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_restore_version(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    node = cortex.restore_version(args["version_id"], agent_id=args.get("agent_id"))
    if node:
        return [TextContent(type="text", text=f"Restored version to memory {node.id}.")]
    return [TextContent(type="text", text="Failed to restore version.")]


async def _handle_list_tags(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    tags = cortex.tag_manager.list_tags(agent_id=args.get("agent_id"))
    if not tags:
        return [TextContent(type="text", text="No tags found.")]
    lines = [f"**{len(tags)} tag(s):**\n"]
    for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
        lines.append(f"- {tag}: {count}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_rename_tag(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    updated = cortex.tag_manager.rename_tag(
        old=args["old_tag"],
        new=args["new_tag"],
        agent_id=args.get("agent_id"),
    )
    return [TextContent(type="text", text=f"Renamed tag in {updated} memories.")]


async def _handle_merge_tags(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    updated = cortex.tag_manager.merge_tags(
        source_tags=_coerce_array(args["source_tags"]) or [],
        target_tag=args["target_tag"],
        agent_id=args.get("agent_id"),
    )
    return [TextContent(type="text", text=f"Merged tags in {updated} memories.")]


async def _handle_delete_tag(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    updated = cortex.tag_manager.delete_tag(
        tag=args["tag"],
        agent_id=args.get("agent_id"),
    )
    return [TextContent(type="text", text=f"Deleted tag from {updated} memories.")]


async def _handle_bulk_delete(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    ids = _coerce_array(args["memory_ids"]) or []
    deleted = cortex.bulk_delete(
        memory_ids=ids,
        soft=not args.get("hard", False),
        agent_id=args.get("agent_id"),
    )
    return [TextContent(type="text", text=f"Deleted {len(deleted)} / {len(ids)} memories.")]


async def _handle_export_memories(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    ids = _coerce_array(args.get("memory_ids"))
    output = cortex.export_memories(
        memory_ids=ids,
        agent_id=args.get("agent_id"),
        fmt=args.get("format", "json"),
    )
    return [TextContent(type="text", text=output[:4000])]


async def _handle_list_threads(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    threads = cortex.list_threads(
        agent_id=args.get("agent_id"),
        limit=args.get("limit", 50),
    )
    if not threads:
        return [TextContent(type="text", text="No threads found.")]
    lines = [f"**{len(threads)} thread(s):**\n"]
    for t in threads:
        lines.append(f"- {t['thread_id']}: {t['memory_count']} memories")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_get_thread_memories(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    results = cortex.get_thread_memories(
        thread_id=args["thread_id"],
        limit=args.get("limit", 100),
    )
    if not results:
        return [TextContent(type="text", text="No memories in thread.")]
    lines = [f"**{len(results)} memory(s) in thread {args['thread_id']}:**\n"]
    for node, _ in results:
        lines.append(f"- [{node.id}] {node.content[:120]}...")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_prune_thread(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    count = cortex.prune_thread(
        thread_id=args["thread_id"],
        agent_id=args.get("agent_id"),
    )
    return [TextContent(type="text", text=f"Soft-deleted {count} memories from thread {args['thread_id']}.")]


# =============================================================================
# Agent Messaging handlers
# =============================================================================

async def _handle_send_message(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    to = args["to"]
    content = args["content"]
    in_reply_to = args.get("in_reply_to")
    tags = _coerce_array(args.get("tags"))

    node = cortex.send_message(
        to=to,
        content=content,
        agent_id=DEFAULT_AGENT_ID,
        in_reply_to=in_reply_to,
        tags=tags if tags else None,
    )

    reply_info = f" (linked to {in_reply_to})" if in_reply_to else ""
    return [TextContent(
        type="text",
        text=(
            f"Message sent to {to}{reply_info}\n"
            f"ID: {node.id}\n"
            f"Tags: {', '.join(node.metadata.tags)}\n"
            f"Content: {content[:200]}{'...' if len(content) > 200 else ''}"
        ),
    )]


async def _handle_check_inbox(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    from_agent = args.get("from_agent")
    limit = args.get("limit", 10)
    since = args.get("since")

    messages = cortex.check_inbox(
        agent_id=DEFAULT_AGENT_ID,
        from_agent=from_agent,
        limit=limit,
        since=since,
    )

    if not messages:
        filter_info = f" from {from_agent}" if from_agent else ""
        since_info = f" since {since}" if since else ""
        return [TextContent(type="text", text=f"No messages{filter_info}{since_info}.")]

    lines = [f"**{len(messages)} message(s):**\n"]
    for msg in messages:
        sender = msg.metadata.agent_id
        ts = msg.created_at.strftime("%Y-%m-%d %H:%M")
        preview = msg.content[:300] + ("..." if len(msg.content) > 300 else "")
        reply_to = msg.metadata.responding_to
        reply_info = f" (reply to {reply_to[0]})" if reply_to else ""
        lines.append(f"---\n**From:** {sender} | **At:** {ts} | **ID:** {msg.id}{reply_info}\n{preview}\n")

    return [TextContent(type="text", text="\n".join(lines))]


# =============================================================================
# MCP Resources
# =============================================================================

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="cerebro://stats",
            name="System Statistics",
            description="Current CerebroCortex system statistics",
            mimeType="application/json",
        ),
    ]

@server.list_resource_templates()
async def list_resource_templates() -> list[ResourceTemplate]:
    return [
        ResourceTemplate(
            uriTemplate="cerebro://memory/{memory_id}",
            name="Memory by ID",
            description="Get a specific memory with full metadata",
            mimeType="application/json",
        ),
        ResourceTemplate(
            uriTemplate="cerebro://episodes/recent",
            name="Recent Episodes",
            description="Get recent episodes",
            mimeType="application/json",
        ),
    ]

@server.read_resource()
async def read_resource(uri) -> str:
    cortex = get_cortex()
    uri_str = str(uri)

    if uri_str == "cerebro://stats":
        stats = cortex.stats()
        return json.dumps(stats, indent=2, default=str)

    if uri_str.startswith("cerebro://memory/"):
        memory_id = uri_str.replace("cerebro://memory/", "")
        node = cortex.get_memory(memory_id)
        if not node:
            return json.dumps({"error": "not found"})
        return json.dumps({
            "id": node.id,
            "content": node.content,
            "type": node.metadata.memory_type.value,
            "layer": node.metadata.layer.value,
            "salience": round(node.metadata.salience, 3),
            "tags": node.metadata.tags,
            "concepts": node.metadata.concepts,
            "created_at": node.created_at.isoformat(),
        }, indent=2)

    if uri_str.startswith("cerebro://episodes"):
        episodes = cortex.list_episodes(limit=10)
        return json.dumps({
            "count": len(episodes),
            "episodes": [
                {
                    "id": ep.id,
                    "title": ep.title,
                    "steps": len(ep.steps),
                    "valence": ep.overall_valence.value if hasattr(ep.overall_valence, "value") else str(ep.overall_valence),
                }
                for ep in episodes
            ],
        }, indent=2)

    return json.dumps({"error": "unknown resource"})


# =============================================================================
# MCP Prompts
# =============================================================================

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="session_handoff",
            description="Generate an end-of-session summary for continuity",
            arguments=[
                PromptArgument(name="session_highlights", description="Key things that happened", required=True),
            ],
        ),
        Prompt(
            name="memory_review",
            description="Review and clean up memories matching a query",
            arguments=[
                PromptArgument(name="query", description="Search query to find memories", required=True),
                PromptArgument(name="max_results", description="Max memories to review (default: 20)", required=False),
            ],
        ),
        Prompt(
            name="context_briefing",
            description="Generate a briefing based on recent activity",
            arguments=[
                PromptArgument(name="focus_area", description="Topic or project to focus on", required=False),
            ],
        ),
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    cortex = get_cortex()
    args = arguments or {}

    if name == "session_handoff":
        highlights = args.get("session_highlights", "")
        intentions = cortex.list_intentions()
        intentions_text = "\n".join(f"- {i.content}" for i in intentions[:5]) or "None"
        return GetPromptResult(
            description="End-of-session handoff",
            messages=[PromptMessage(
                role="user",
                content=TextContent(type="text", text=(
                    f"Create a session handoff note based on:\n\n{highlights}\n\n"
                    f"Pending intentions:\n{intentions_text}\n\n"
                    f"Use session_save to store a note with key_discoveries, "
                    f"unfinished_business, and if_disoriented fields."
                )),
            )],
        )

    elif name == "memory_review":
        query = args.get("query", "")
        max_results = int(args.get("max_results", "20"))
        results = cortex.recall(query=query, top_k=max_results)
        memories_text = "\n\n".join(
            f"ID: {n.id}\nType: {n.metadata.memory_type.value} | Salience: {n.metadata.salience:.2f}\n"
            f"Content: {n.content[:300]}\nTags: {', '.join(n.metadata.tags)}"
            for n, score in results
        )
        return GetPromptResult(
            description=f"Review {len(results)} memories matching '{query}'",
            messages=[PromptMessage(
                role="user",
                content=TextContent(type="text", text=(
                    f"Review these {len(results)} memories and suggest:\n"
                    f"1. Which to delete (low quality, outdated, duplicates)\n"
                    f"2. Which salience to adjust\n"
                    f"3. Which to link together\n\n{memories_text}\n\n"
                    f"Use delete_memory, update_memory, and associate tools."
                )),
            )],
        )

    elif name == "context_briefing":
        focus = args.get("focus_area", "")
        query = focus or "recent work and progress"
        results = cortex.recall(query=query, top_k=10)
        intentions = cortex.list_intentions()
        context_text = "\n".join(
            f"- [{n.metadata.memory_type.value}] {n.content[:150]}"
            for n, _ in results
        )
        intentions_text = "\n".join(f"- {i.content}" for i in intentions[:5]) or "None"
        return GetPromptResult(
            description=f"Context briefing{' on ' + focus if focus else ''}",
            messages=[PromptMessage(
                role="user",
                content=TextContent(type="text", text=(
                    f"Briefing on{' ' + focus if focus else ' recent activity'}:\n\n"
                    f"Relevant memories:\n{context_text}\n\n"
                    f"Pending intentions:\n{intentions_text}\n\n"
                    f"Summarize the current state and suggest next steps."
                )),
            )],
        )

    raise ValueError(f"Unknown prompt: {name}")


# =============================================================================
# Main entry point
# =============================================================================

async def main():
    logger.info(f"Starting CerebroCortex MCP Server v{MCP_SERVER_VERSION}...")

    try:
        from cerebro.settings import load_on_startup
        load_on_startup()
        logger.info("Settings loaded from disk")
    except Exception as e:
        logger.warning(f"Failed to load settings: {e}")

    try:
        ctx = get_cortex()
        s = ctx.stats()
        logger.info(
            f"Cortex initialized: {s['nodes']} memories, "
            f"{s['links']} links, {s['episodes']} episodes"
        )
    except Exception as e:
        logger.error(f"Failed to initialize cortex: {e}", exc_info=True)

    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main_sync():
    """Synchronous entry point for console_scripts."""
    from cerebro._init_dirs import ensure_data_dirs
    ensure_data_dirs()
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
