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
from mcp.types import Tool, TextContent

from cerebro.config import MCP_SERVER_NAME, MCP_SERVER_VERSION
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


# =============================================================================
# Tool Schemas
# =============================================================================

TOOL_SCHEMAS: dict[str, dict] = {
    # -----------------------------------------------------------------
    # Core: remember, recall, associate
    # -----------------------------------------------------------------
    "remember": {
        "description": "Store a memory in CerebroCortex. The memory goes through gating (dedup/noise filter), emotional analysis, concept extraction, and auto-linking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content to store"},
                "memory_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural", "affective", "prospective", "schematic"],
                    "description": "Memory type (auto-classified if omitted)",
                },
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
                "salience": {"type": "number", "description": "Importance 0-1 (auto-estimated if omitted)"},
                "agent_id": {"type": "string", "description": "Agent storing this memory"},
                "session_id": {"type": "string", "description": "Current session ID"},
                "visibility": {
                    "type": "string",
                    "enum": ["private", "shared", "thread"],
                    "description": "Sharing scope",
                },
                "context_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of memories active in current context (for auto-linking)",
                },
            },
            "required": ["content"],
        },
    },
    "recall": {
        "description": "Search and retrieve memories using spreading activation + ACT-R/FSRS scoring. Provides richer results than simple vector search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {"type": "integer", "description": "Max results (default: 10)"},
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["episodic", "semantic", "procedural", "affective", "prospective", "schematic"]},
                    "description": "Filter by memory type",
                },
                "agent_id": {"type": "string", "description": "Filter by agent"},
                "min_salience": {"type": "number", "description": "Minimum salience threshold"},
                "context_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Seed memory IDs for spreading activation",
                },
            },
            "required": ["query"],
        },
    },
    "associate": {
        "description": "Create a typed associative link between two memories.",
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
    # Episodes
    # -----------------------------------------------------------------
    "episode_start": {
        "description": "Begin recording a new episode (temporal sequence of events).",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Episode title"},
                "session_id": {"type": "string", "description": "Session this episode belongs to"},
                "agent_id": {"type": "string", "description": "Agent recording the episode"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
        },
    },
    "episode_add_step": {
        "description": "Add a memory as a step in the current episode. Creates temporal links between sequential steps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string", "description": "Episode to add to"},
                "memory_id": {"type": "string", "description": "Memory being added"},
                "role": {
                    "type": "string",
                    "enum": ["event", "context", "outcome", "reflection"],
                    "description": "Role of this step",
                },
            },
            "required": ["episode_id", "memory_id"],
        },
    },
    "episode_end": {
        "description": "End an episode, setting its summary and emotional valence.",
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
    # Sessions (backward-compatible with Neo-Cortex)
    # -----------------------------------------------------------------
    "session_save": {
        "description": "Save a session note for future instances. Use at session end for continuity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_summary": {"type": "string", "description": "What happened this session"},
                "key_discoveries": {"type": "array", "items": {"type": "string"}, "description": "Important findings"},
                "unfinished_business": {"type": "array", "items": {"type": "string"}, "description": "Tasks to continue"},
                "if_disoriented": {"type": "array", "items": {"type": "string"}, "description": "Orientation instructions"},
                "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "session_type": {"type": "string", "enum": ["orientation", "technical", "emotional", "task"]},
            },
            "required": ["session_summary"],
        },
    },
    "session_recall": {
        "description": "Retrieve session notes from previous instances.",
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
        "description": "Register a new agent in the memory system.",
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
        "description": "List all registered agents.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },

    # -----------------------------------------------------------------
    # Health & Stats
    # -----------------------------------------------------------------
    "memory_health": {
        "description": "Get memory system health report including graph stats, layer distribution, and engine status.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "memory_graph_stats": {
        "description": "Get detailed graph statistics: node counts by type/layer, link counts by type, igraph metrics.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "memory_neighbors": {
        "description": "Get the neighbors of a memory in the associative graph.",
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
        "description": "Get comprehensive CerebroCortex system statistics.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },

    # -----------------------------------------------------------------
    # Dream Engine
    # -----------------------------------------------------------------
    "dream_run": {
        "description": "Run a dream consolidation cycle. Processes unconsolidated episodes through 6 phases: SWS replay, pattern extraction, schema formation, emotional reprocessing, pruning, REM recombination.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_llm_calls": {"type": "integer", "description": "Max LLM calls for this cycle (default: 20)"},
            },
            "required": [],
        },
    },
    "dream_status": {
        "description": "Get the status and last report from the Dream Engine.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },

    # -----------------------------------------------------------------
    # Backward-compatible Neo-Cortex aliases
    # -----------------------------------------------------------------
    "memory_store": {
        "description": "[Neo-Cortex compat] Store a memory. Alias for 'remember'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content"},
                "visibility": {"type": "string", "enum": ["private", "shared", "thread"]},
                "message_type": {"type": "string", "description": "Type of message"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "conversation_thread": {"type": "string"},
                "related_agents": {"type": "array", "items": {"type": "string"}},
                "responding_to": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["content"],
        },
    },
    "memory_search": {
        "description": "[Neo-Cortex compat] Search memories. Alias for 'recall'.",
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
        # Episodes
        # =============================================================
        elif name == "episode_start":
            return await _handle_episode_start(cortex, arguments)

        elif name == "episode_add_step":
            return await _handle_episode_add_step(cortex, arguments)

        elif name == "episode_end":
            return await _handle_episode_end(cortex, arguments)

        # =============================================================
        # Sessions
        # =============================================================
        elif name == "session_save":
            return await _handle_session_save(cortex, arguments)

        elif name == "session_recall":
            return await _handle_session_recall(cortex, arguments)

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

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool error [{name}]: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# =============================================================================
# Handler implementations
# =============================================================================

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
        tags=args.get("tags"),
        salience=args.get("salience"),
        agent_id=args.get("agent_id", "CLAUDE"),
        session_id=args.get("session_id"),
        visibility=visibility,
        context_ids=args.get("context_ids") or args.get("responding_to"),
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
        memory_types = [MemoryType(t) for t in args["memory_types"]]

    results = cortex.recall(
        query=args["query"],
        top_k=args.get("top_k", args.get("n_results", 10)),
        memory_types=memory_types,
        agent_id=args.get("agent_id", args.get("agent_filter")),
        min_salience=args.get("min_salience", 0.0),
        context_ids=args.get("context_ids"),
    )

    if not results:
        return [TextContent(type="text", text="No memories found.")]

    lines = [f"**Found {len(results)} memories:**\n"]
    for i, (node, score) in enumerate(results, 1):
        content_preview = node.content[:150] + "..." if len(node.content) > 150 else node.content
        lines.append(
            f"{i}. [{node.metadata.memory_type.value}] (score: {score:.3f}, "
            f"salience: {node.metadata.salience:.2f}) "
            f"{content_preview}"
        )

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


async def _handle_episode_start(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    episode = cortex.episode_start(
        title=args.get("title"),
        session_id=args.get("session_id"),
        agent_id=args.get("agent_id", "CLAUDE"),
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


async def _handle_session_save(cortex: CerebroCortex, args: dict) -> list[TextContent]:
    # Build session note content
    parts = [f"SESSION SUMMARY: {args['session_summary']}"]

    if args.get("key_discoveries"):
        parts.append("\nKEY DISCOVERIES:")
        for d in args["key_discoveries"]:
            parts.append(f"  - {d}")

    if args.get("unfinished_business"):
        parts.append("\nUNFINISHED BUSINESS:")
        for u in args["unfinished_business"]:
            parts.append(f"  - {u}")

    if args.get("if_disoriented"):
        parts.append("\nIF DISORIENTED:")
        for o in args["if_disoriented"]:
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

    return [TextContent(
        type="text",
        text=f"Session note saved (ID: {node.id}, priority: {priority})",
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
    for neighbor_id, weight, link_type in neighbors:
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

    report = dream.run_cycle()
    phase_summary = ", ".join(
        f"{p.phase.value}({'ok' if p.success else 'FAIL'})" for p in report.phases
    )

    return [TextContent(
        type="text",
        text=(
            f"**Dream Cycle Complete**\n"
            f"Duration: {report.total_duration_seconds:.1f}s\n"
            f"Episodes consolidated: {report.episodes_consolidated}\n"
            f"LLM calls: {report.total_llm_calls}\n"
            f"Phases: {phase_summary}\n"
            f"Success: {report.success}"
        ),
    )]


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
# Main entry point
# =============================================================================

async def main():
    logger.info(f"Starting CerebroCortex MCP Server v{MCP_SERVER_VERSION}...")

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


if __name__ == "__main__":
    asyncio.run(main())
