"""CerebroCortex REST API Server.

FastAPI server exposing the full CerebroCortex memory system over HTTP.
Port 8767 by default (Neo-Cortex uses 8766).

Usage:
    python -m cerebro.interfaces.api_server
    ./cerebro-api

Endpoints:
    GET  /          API info
    GET  /health    Health check
    GET  /stats     Full statistics
    GET  /ui        Web dashboard
    GET  /q/{query} Quick search

    POST /remember      Store a memory
    POST /recall        Search memories
    POST /associate     Create link

    POST /episodes/start       Start episode
    POST /episodes/{id}/step   Add step
    POST /episodes/{id}/end    End episode
    GET  /episodes             List recent episodes
    GET  /episodes/{id}        Get episode by ID
    GET  /episodes/{id}/memories  Get episode memories

    POST /intentions           Store intention
    GET  /intentions           List pending intentions
    POST /intentions/{id}/resolve  Resolve intention

    POST /sessions/save   Save session note
    GET  /sessions        Get session notes

    GET  /agents          List agents
    POST /agents          Register agent

    GET  /memory/health          Health report
    GET  /graph/stats            Graph stats
    GET  /graph/data             Graph data for visualization
    GET  /graph/neighbors/{id}   Neighbors
    GET  /graph/path/{src}/{tgt} Find shortest path
    GET  /graph/common/{a}/{b}   Common neighbors

    POST /schemas                Create schema
    GET  /schemas                List schemas
    GET  /schemas/match          Find matching schemas
    GET  /schemas/{id}/sources   Get schema sources

    POST /procedures             Store procedure
    GET  /procedures             List procedures
    GET  /procedures/match       Find relevant procedures
    POST /procedures/{id}/outcome  Record procedure outcome

    GET  /emotions/summary       Emotional valence breakdown
"""

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from cerebro.config import (
    API_HOST,
    API_PORT,
    MCP_SERVER_NAME,
    MCP_SERVER_VERSION,
)
from cerebro.cortex import CerebroCortex
from cerebro.models.agent import AgentProfile
from cerebro.types import EmotionalValence, LinkType, MemoryType, Visibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("cerebro-api")

# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(
    title="CerebroCortex API",
    description="Brain-analogous AI memory system with associative networks, "
    "ACT-R + FSRS activation, and spreading activation.",
    version=MCP_SERVER_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton cortex instance
_cortex: Optional[CerebroCortex] = None
_dream_engine = None

WEB_DIR = Path(__file__).parent.parent.parent.parent / "web"


def get_cortex() -> CerebroCortex:
    global _cortex
    if _cortex is None:
        _cortex = CerebroCortex()
        _cortex.initialize()
        logger.info("CerebroCortex initialized")
    return _cortex


# =============================================================================
# Request/Response models
# =============================================================================

class RememberRequest(BaseModel):
    content: str
    memory_type: Optional[str] = None
    tags: Optional[list[str]] = None
    salience: Optional[float] = None
    agent_id: str = "CLAUDE"
    session_id: Optional[str] = None
    visibility: str = "shared"
    context_ids: Optional[list[str]] = None

class RecallRequest(BaseModel):
    query: str
    top_k: int = 10
    memory_types: Optional[list[str]] = None
    agent_id: Optional[str] = None
    min_salience: float = 0.0
    context_ids: Optional[list[str]] = None
    conversation_thread: Optional[str] = None

class AssociateRequest(BaseModel):
    source_id: str
    target_id: str
    link_type: str
    weight: float = 0.5
    evidence: Optional[str] = None

class EpisodeStartRequest(BaseModel):
    title: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: str = "CLAUDE"

class EpisodeStepRequest(BaseModel):
    memory_id: str
    role: str = "event"

class EpisodeEndRequest(BaseModel):
    summary: Optional[str] = None
    valence: str = "neutral"

class SessionSaveRequest(BaseModel):
    session_summary: str
    key_discoveries: Optional[list[str]] = None
    unfinished_business: Optional[list[str]] = None
    if_disoriented: Optional[list[str]] = None
    priority: str = "MEDIUM"
    session_type: str = "orientation"

class RegisterAgentRequest(BaseModel):
    agent_id: str
    display_name: str
    generation: int = 0
    lineage: str = ""
    specialization: str = ""
    origin_story: Optional[str] = None
    color: str = "#888888"
    symbol: str = "A"

class IntentionRequest(BaseModel):
    content: str
    tags: Optional[list[str]] = None
    agent_id: str = "CLAUDE"
    salience: float = 0.7

class UpdateMemoryRequest(BaseModel):
    content: Optional[str] = None
    tags: Optional[list[str]] = None
    salience: Optional[float] = None
    visibility: Optional[str] = None

class CreateSchemaRequest(BaseModel):
    content: str
    source_ids: list[str]
    tags: Optional[list[str]] = None
    agent_id: str = "CLAUDE"

class StoreProcedureRequest(BaseModel):
    content: str
    tags: Optional[list[str]] = None
    derived_from: Optional[list[str]] = None
    agent_id: str = "CLAUDE"

class ProcedureOutcomeRequest(BaseModel):
    success: bool


# =============================================================================
# Info & health
# =============================================================================

@app.get("/")
async def root():
    """API info and version."""
    return {
        "name": MCP_SERVER_NAME,
        "version": MCP_SERVER_VERSION,
        "description": "CerebroCortex - Brain-analogous AI memory system",
        "endpoints": [
            "GET /health", "GET /stats", "GET /ui", "GET /q/{query}",
            "POST /remember", "POST /recall", "POST /associate",
            "GET /memory/{id}", "DELETE /memory/{id}", "PATCH /memory/{id}", "POST /memory/{id}/share",
            "POST /episodes/start", "POST /episodes/{id}/step", "POST /episodes/{id}/end",
            "GET /episodes", "GET /episodes/{id}", "GET /episodes/{id}/memories",
            "POST /intentions", "GET /intentions", "POST /intentions/{id}/resolve",
            "POST /sessions/save", "GET /sessions",
            "GET /agents", "POST /agents",
            "GET /memory/health", "GET /graph/stats", "GET /graph/data",
            "GET /graph/neighbors/{id}",
            "GET /graph/path/{src}/{tgt}", "GET /graph/common/{a}/{b}",
            "POST /schemas", "GET /schemas", "GET /schemas/match",
            "GET /schemas/{id}/sources",
            "POST /procedures", "GET /procedures", "GET /procedures/match",
            "POST /procedures/{id}/outcome",
            "GET /emotions/summary",
        ],
    }


@app.get("/health")
async def health():
    """Health check."""
    try:
        ctx = get_cortex()
        s = ctx.stats()
        return {
            "status": "healthy",
            "memories": s["nodes"],
            "links": s["links"],
            "episodes": s["episodes"],
            "initialized": s["initialized"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/stats")
async def stats():
    """Full system statistics."""
    ctx = get_cortex()
    return ctx.stats()


@app.get("/ui")
async def dashboard():
    """Serve the web dashboard."""
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(index, media_type="text/html")


@app.get("/q/{query:path}")
async def quick_search(query: str, n: int = Query(5, ge=1, le=50)):
    """Quick search convenience endpoint."""
    ctx = get_cortex()
    results = ctx.recall(query=query, top_k=n)
    return {
        "query": query,
        "count": len(results),
        "results": [
            {
                "id": node.id,
                "content": node.content,
                "type": node.metadata.memory_type.value,
                "salience": node.metadata.salience,
                "score": round(score, 4),
                "tags": node.metadata.tags,
            }
            for node, score in results
        ],
    }


# =============================================================================
# Core memory operations
# =============================================================================

@app.post("/remember")
async def remember(req: RememberRequest):
    """Store a memory through the full encoding pipeline."""
    ctx = get_cortex()

    memory_type = None
    if req.memory_type:
        try:
            memory_type = MemoryType(req.memory_type)
        except ValueError:
            raise HTTPException(400, f"Invalid memory_type: {req.memory_type}")

    try:
        visibility = Visibility(req.visibility)
    except ValueError:
        raise HTTPException(400, f"Invalid visibility: {req.visibility}")

    node = ctx.remember(
        content=req.content,
        memory_type=memory_type,
        tags=req.tags,
        salience=req.salience,
        agent_id=req.agent_id,
        session_id=req.session_id,
        visibility=visibility,
        context_ids=req.context_ids,
    )

    if node is None:
        return {"stored": False, "reason": "gated_out"}

    return {
        "stored": True,
        "id": node.id,
        "type": node.metadata.memory_type.value,
        "layer": node.metadata.layer.value,
        "salience": round(node.metadata.salience, 3),
        "valence": node.metadata.valence.value if hasattr(node.metadata.valence, "value") else str(node.metadata.valence),
        "concepts": node.metadata.concepts[:10],
        "links": node.link_count,
    }


@app.post("/recall")
async def recall(req: RecallRequest):
    """Search and retrieve memories with spreading activation + ACT-R/FSRS scoring."""
    ctx = get_cortex()

    memory_types = None
    if req.memory_types:
        try:
            memory_types = [MemoryType(t) for t in req.memory_types]
        except ValueError as e:
            raise HTTPException(400, f"Invalid memory_type: {e}")

    results = ctx.recall(
        query=req.query,
        top_k=req.top_k,
        memory_types=memory_types,
        agent_id=req.agent_id,
        min_salience=req.min_salience,
        context_ids=req.context_ids,
        conversation_thread=req.conversation_thread,
    )

    return {
        "query": req.query,
        "count": len(results),
        "results": [
            {
                "id": node.id,
                "content": node.content,
                "type": node.metadata.memory_type.value,
                "layer": node.metadata.layer.value,
                "salience": round(node.metadata.salience, 3),
                "valence": node.metadata.valence.value if hasattr(node.metadata.valence, "value") else str(node.metadata.valence),
                "score": round(score, 4),
                "tags": node.metadata.tags,
                "concepts": node.metadata.concepts[:5],
                "created_at": node.created_at.isoformat(),
            }
            for node, score in results
        ],
    }


@app.post("/associate")
async def associate(req: AssociateRequest):
    """Create an associative link between two memories."""
    ctx = get_cortex()

    try:
        link_type = LinkType(req.link_type)
    except ValueError:
        raise HTTPException(400, f"Invalid link_type: {req.link_type}")

    link_id = ctx.associate(
        source_id=req.source_id,
        target_id=req.target_id,
        link_type=link_type,
        weight=req.weight,
        evidence=req.evidence,
    )

    if link_id is None:
        raise HTTPException(404, "One or both memory IDs not found")

    return {
        "link_id": link_id,
        "source_id": req.source_id,
        "target_id": req.target_id,
        "link_type": req.link_type,
        "weight": req.weight,
    }


# =============================================================================
# Episodes
# =============================================================================

@app.post("/episodes/start")
async def episode_start(req: EpisodeStartRequest):
    """Start recording a new episode."""
    ctx = get_cortex()
    episode = ctx.episode_start(
        title=req.title,
        session_id=req.session_id,
        agent_id=req.agent_id,
    )
    return {
        "id": episode.id,
        "title": episode.title,
        "started_at": episode.started_at.isoformat() if episode.started_at else None,
    }


@app.post("/episodes/{episode_id}/step")
async def episode_add_step(episode_id: str, req: EpisodeStepRequest):
    """Add a memory as a step in the episode."""
    ctx = get_cortex()
    step = ctx.episodes.add_step(
        episode_id=episode_id,
        memory_id=req.memory_id,
        role=req.role,
    )
    if step is None:
        raise HTTPException(404, "Episode not found")

    return {
        "episode_id": episode_id,
        "memory_id": req.memory_id,
        "position": step.position,
        "role": step.role,
    }


@app.post("/episodes/{episode_id}/end")
async def episode_end(episode_id: str, req: EpisodeEndRequest):
    """End an episode with summary and valence."""
    ctx = get_cortex()

    try:
        valence = EmotionalValence(req.valence)
    except ValueError:
        valence = EmotionalValence.NEUTRAL

    episode = ctx.episode_end(
        episode_id=episode_id,
        summary=req.summary,
        valence=valence,
    )
    if episode is None:
        raise HTTPException(404, "Episode not found")

    return {
        "id": episode.id,
        "title": episode.title,
        "steps": len(episode.steps),
        "valence": episode.overall_valence.value,
        "ended_at": episode.ended_at.isoformat() if episode.ended_at else None,
    }



@app.get("/episodes")
async def list_episodes(
    limit: int = Query(10, ge=1, le=100),
    agent_id: Optional[str] = None,
):
    """List recent episodes."""
    ctx = get_cortex()
    episodes = ctx.list_episodes(limit=limit, agent_id=agent_id)
    return {
        "count": len(episodes),
        "episodes": [
            {
                "id": ep.id,
                "title": ep.title,
                "steps": len(ep.steps),
                "valence": ep.overall_valence.value,
                "started_at": ep.started_at.isoformat() if ep.started_at else None,
                "ended_at": ep.ended_at.isoformat() if ep.ended_at else None,
                "consolidated": ep.consolidated,
            }
            for ep in episodes
        ],
    }


@app.get("/episodes/{episode_id}")
async def get_episode(episode_id: str):
    """Get a single episode by ID."""
    ctx = get_cortex()
    episode = ctx.get_episode(episode_id)
    if episode is None:
        raise HTTPException(404, f"Episode not found: {episode_id}")
    return {
        "id": episode.id,
        "title": episode.title,
        "steps": len(episode.steps),
        "valence": episode.overall_valence.value,
        "started_at": episode.started_at.isoformat() if episode.started_at else None,
        "ended_at": episode.ended_at.isoformat() if episode.ended_at else None,
        "consolidated": episode.consolidated,
    }


@app.get("/episodes/{episode_id}/memories")
async def get_episode_memories(episode_id: str):
    """Get all memories in an episode, ordered by position."""
    ctx = get_cortex()
    memories = ctx.get_episode_memories(episode_id)
    return {
        "episode_id": episode_id,
        "count": len(memories),
        "memories": [
            {
                "id": node.id,
                "content": node.content,
                "type": node.metadata.memory_type.value,
                "salience": round(node.metadata.salience, 3),
                "tags": node.metadata.tags,
                "created_at": node.created_at.isoformat(),
            }
            for node in memories
        ],
    }


# =============================================================================
# Sessions
# =============================================================================

@app.post("/sessions/save")
async def session_save(req: SessionSaveRequest):
    """Save a session note for continuity."""
    ctx = get_cortex()

    parts = [f"SESSION SUMMARY: {req.session_summary}"]
    if req.key_discoveries:
        parts.append("\nKEY DISCOVERIES:")
        for d in req.key_discoveries:
            parts.append(f"  - {d}")
    if req.unfinished_business:
        parts.append("\nUNFINISHED BUSINESS:")
        for u in req.unfinished_business:
            parts.append(f"  - {u}")
    if req.if_disoriented:
        parts.append("\nIF DISORIENTED:")
        for o in req.if_disoriented:
            parts.append(f"  - {o}")

    content = "\n".join(parts)

    node = ctx.remember(
        content=content,
        memory_type=MemoryType.EPISODIC,
        tags=[
            "session_note",
            f"priority:{req.priority}",
            f"session_type:{req.session_type}",
        ],
        salience={"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.4}.get(req.priority, 0.7),
    )

    if node is None:
        raise HTTPException(400, "Session note gated out (duplicate?)")

    return {
        "id": node.id,
        "priority": req.priority,
        "session_type": req.session_type,
    }


@app.get("/sessions")
async def session_recall(
    lookback_hours: int = Query(168, ge=1),
    priority: Optional[str] = None,
    session_type: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
):
    """Retrieve session notes."""
    ctx = get_cortex()
    since = datetime.now() - timedelta(hours=lookback_hours)
    nodes = ctx.graph.get_nodes_since(since)

    sessions = []
    for node in nodes:
        if "session_note" not in node.metadata.tags:
            continue
        if priority and f"priority:{priority}" not in node.metadata.tags:
            continue
        if session_type and f"session_type:{session_type}" not in node.metadata.tags:
            continue
        sessions.append(node)

    sessions.sort(key=lambda n: n.created_at, reverse=True)
    sessions = sessions[:limit]

    return {
        "count": len(sessions),
        "sessions": [
            {
                "id": s.id,
                "content": s.content,
                "priority": next(
                    (t.split(":")[1] for t in s.metadata.tags if t.startswith("priority:")),
                    "MEDIUM",
                ),
                "session_type": next(
                    (t.split(":")[1] for t in s.metadata.tags if t.startswith("session_type:")),
                    "orientation",
                ),
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions
        ],
    }


# =============================================================================
# Intentions (prospective memory)
# =============================================================================

@app.post("/intentions")
async def store_intention(req: IntentionRequest):
    """Store a prospective memory (future intention / TODO)."""
    ctx = get_cortex()
    node = ctx.store_intention(
        content=req.content,
        tags=req.tags,
        agent_id=req.agent_id,
        salience=req.salience,
    )
    return {
        "id": node.id,
        "content": node.content,
        "salience": round(node.metadata.salience, 3),
        "tags": node.metadata.tags,
        "agent_id": node.metadata.agent_id,
        "created_at": node.created_at.isoformat(),
    }


@app.get("/intentions")
async def list_intentions(
    agent_id: Optional[str] = None,
    min_salience: float = Query(0.3, ge=0.0, le=1.0),
):
    """List pending intentions."""
    ctx = get_cortex()
    intentions = ctx.list_intentions(agent_id=agent_id, min_salience=min_salience)
    return {
        "count": len(intentions),
        "intentions": [
            {
                "id": node.id,
                "content": node.content,
                "salience": round(node.metadata.salience, 3),
                "tags": node.metadata.tags,
                "agent_id": node.metadata.agent_id,
                "created_at": node.created_at.isoformat(),
            }
            for node in intentions
        ],
    }


@app.post("/intentions/{memory_id}/resolve")
async def resolve_intention(memory_id: str):
    """Resolve a pending intention."""
    ctx = get_cortex()
    success = ctx.resolve_intention(memory_id)
    if not success:
        raise HTTPException(404, f"Intention not found or already resolved: {memory_id}")
    return {"resolved": True, "id": memory_id}


# =============================================================================
# Agents
# =============================================================================

@app.get("/agents")
async def list_agents():
    """List all registered agents."""
    ctx = get_cortex()
    agents = ctx.graph.list_agents()
    return {
        "count": len(agents),
        "agents": [
            {
                "id": a.id,
                "display_name": a.display_name,
                "generation": a.generation,
                "lineage": a.lineage,
                "specialization": a.specialization,
                "color": a.color,
                "symbol": a.symbol,
            }
            for a in agents
        ],
    }


@app.post("/agents")
async def register_agent(req: RegisterAgentRequest):
    """Register a new agent."""
    ctx = get_cortex()
    profile = AgentProfile(
        id=req.agent_id,
        display_name=req.display_name,
        generation=req.generation,
        lineage=req.lineage,
        specialization=req.specialization,
        origin_story=req.origin_story,
        color=req.color,
        symbol=req.symbol,
    )
    ctx.graph.register_agent(profile)
    return {
        "registered": True,
        "agent_id": profile.id,
        "display_name": profile.display_name,
    }


# =============================================================================
# Health & graph stats
# =============================================================================

@app.get("/memory/health")
async def memory_health():
    """Memory system health report."""
    ctx = get_cortex()
    s = ctx.stats()

    promotions = ctx.executive.run_promotion_sweep()
    intentions = ctx.executive.get_pending_intentions()

    return {
        "memories": s["nodes"],
        "links": s["links"],
        "episodes": s["episodes"],
        "schemas": s["schemas"],
        "by_type": s.get("memory_types", {}),
        "by_layer": s.get("layers", {}),
        "promotions": promotions,
        "pending_intentions": len(intentions),
    }


@app.get("/graph/stats")
async def graph_stats():
    """Detailed graph statistics."""
    ctx = get_cortex()
    s = ctx.stats()
    return {
        "nodes": s["nodes"],
        "links": s["links"],
        "igraph_vertices": s["igraph_vertices"],
        "igraph_edges": s["igraph_edges"],
        "link_types": s.get("link_types", {}),
        "memory_types": s.get("memory_types", {}),
        "layers": s.get("layers", {}),
    }


@app.get("/graph/data")
async def graph_data(
    limit: int = Query(200, ge=1, le=2000),
    min_salience: float = Query(0.0, ge=0.0, le=1.0),
):
    """Get graph data (nodes + links) for visualization."""
    ctx = get_cortex()

    # Get nodes from SQLite, ordered by salience descending
    rows = ctx.graph.conn.execute(
        "SELECT id, content, memory_type, layer, salience, valence, "
        "arousal, tags_json, concepts_json, agent_id, access_count, "
        "created_at FROM memory_nodes "
        "WHERE salience >= ? "
        "ORDER BY salience DESC LIMIT ?",
        (min_salience, limit),
    ).fetchall()

    node_ids = set()
    nodes = []
    for r in rows:
        node_ids.add(r["id"])
        nodes.append({
            "id": r["id"],
            "content": r["content"][:300],
            "type": r["memory_type"],
            "layer": r["layer"],
            "salience": round(r["salience"], 3),
            "valence": r["valence"],
            "arousal": round(r["arousal"], 3),
            "tags": json.loads(r["tags_json"]) if r["tags_json"] else [],
            "concepts": json.loads(r["concepts_json"]) if r["concepts_json"] else [],
            "agent_id": r["agent_id"],
            "access_count": r["access_count"],
            "created_at": r["created_at"],
        })

    # Get links between the visible nodes
    links = []
    if node_ids:
        placeholders = ",".join("?" * len(node_ids))
        link_rows = ctx.graph.conn.execute(
            f"SELECT source_id, target_id, link_type, weight FROM associative_links "
            f"WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})",
            list(node_ids) + list(node_ids),
        ).fetchall()
        for lr in link_rows:
            links.append({
                "source": lr["source_id"],
                "target": lr["target_id"],
                "type": lr["link_type"],
                "weight": round(lr["weight"], 3),
            })

    return {"nodes": nodes, "links": links}


@app.get("/graph/neighbors/{memory_id}")
async def graph_neighbors(
    memory_id: str,
    max_results: int = Query(10, ge=1, le=100),
):
    """Get neighbors of a memory in the associative graph."""
    ctx = get_cortex()
    neighbors = ctx.links.get_strongest_connections(memory_id, top_n=max_results)

    return {
        "memory_id": memory_id,
        "count": len(neighbors),
        "neighbors": [
            {
                "id": nid,
                "weight": round(w, 3),
                "link_type": lt,
                "content": (ctx.graph.get_node(nid).content[:200] if ctx.graph.get_node(nid) else ""),
            }
            for nid, w, lt in neighbors
        ],
    }


# =============================================================================
# Graph exploration (path finding, common neighbors)
# =============================================================================

@app.get("/graph/path/{source_id}/{target_id}")
async def graph_find_path(source_id: str, target_id: str):
    """Find shortest path between two memories in the associative graph."""
    ctx = get_cortex()
    path = ctx.find_path(source_id, target_id)
    if path is None:
        raise HTTPException(404, f"No path found between {source_id} and {target_id}")
    return {
        "source": source_id,
        "target": target_id,
        "path": path,
        "length": len(path),
    }


@app.get("/graph/common/{id_a}/{id_b}")
async def graph_common_neighbors(id_a: str, id_b: str):
    """Find memories connected to both A and B."""
    ctx = get_cortex()
    common_ids = ctx.get_common_neighbors(id_a, id_b)
    results = []
    for cid in common_ids:
        node = ctx.graph.get_node(cid)
        results.append({
            "id": cid,
            "content": node.content[:200] if node else "",
            "type": node.metadata.memory_type.value if node else "unknown",
        })
    return {
        "id_a": id_a,
        "id_b": id_b,
        "count": len(results),
        "common": results,
    }


# =============================================================================
# Schemas (schematic memory / neocortex)
# =============================================================================

@app.post("/schemas")
async def create_schema(req: CreateSchemaRequest):
    """Create an abstract schema from source memories."""
    ctx = get_cortex()
    node = ctx.create_schema(
        content=req.content,
        source_ids=req.source_ids,
        tags=req.tags,
        agent_id=req.agent_id,
    )
    return {
        "id": node.id,
        "content": node.content,
        "type": node.metadata.memory_type.value,
        "salience": round(node.metadata.salience, 3),
        "tags": node.metadata.tags,
        "source_ids": req.source_ids,
        "created_at": node.created_at.isoformat(),
    }


@app.get("/schemas")
async def list_schemas(agent_id: Optional[str] = None):
    """List all schematic memories."""
    ctx = get_cortex()
    schemas = ctx.list_schemas(agent_id=agent_id)
    return {
        "count": len(schemas),
        "schemas": [
            {
                "id": node.id,
                "content": node.content[:300],
                "salience": round(node.metadata.salience, 3),
                "tags": node.metadata.tags,
                "concepts": node.metadata.concepts[:10],
                "agent_id": node.metadata.agent_id,
                "created_at": node.created_at.isoformat(),
            }
            for node in schemas
        ],
    }


@app.get("/schemas/match")
async def match_schemas(
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    concepts: Optional[str] = Query(None, description="Comma-separated concepts"),
):
    """Find schemas matching given tags or concepts."""
    ctx = get_cortex()
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    concept_list = [c.strip() for c in concepts.split(",")] if concepts else None
    schemas = ctx.find_matching_schemas(tags=tag_list, concepts=concept_list)
    return {
        "count": len(schemas),
        "schemas": [
            {
                "id": node.id,
                "content": node.content[:300],
                "salience": round(node.metadata.salience, 3),
                "tags": node.metadata.tags,
                "concepts": node.metadata.concepts[:10],
                "created_at": node.created_at.isoformat(),
            }
            for node in schemas
        ],
    }


@app.get("/schemas/{schema_id}/sources")
async def get_schema_sources(schema_id: str):
    """Get the source memory IDs that a schema was derived from."""
    ctx = get_cortex()
    source_ids = ctx.get_schema_sources(schema_id)
    return {
        "schema_id": schema_id,
        "count": len(source_ids),
        "source_ids": source_ids,
    }


# =============================================================================
# Procedures (procedural memory / cerebellum)
# =============================================================================

@app.post("/procedures")
async def store_procedure(req: StoreProcedureRequest):
    """Store a procedural memory (strategy/workflow)."""
    ctx = get_cortex()
    node = ctx.store_procedure(
        content=req.content,
        tags=req.tags,
        derived_from=req.derived_from,
        agent_id=req.agent_id,
    )
    return {
        "id": node.id,
        "content": node.content,
        "type": node.metadata.memory_type.value,
        "salience": round(node.metadata.salience, 3),
        "tags": node.metadata.tags,
        "agent_id": node.metadata.agent_id,
        "created_at": node.created_at.isoformat(),
    }


@app.get("/procedures")
async def list_procedures(
    agent_id: Optional[str] = None,
    min_salience: float = Query(0.0, ge=0.0, le=1.0),
):
    """List all procedural memories."""
    ctx = get_cortex()
    procedures = ctx.list_procedures(agent_id=agent_id, min_salience=min_salience)
    return {
        "count": len(procedures),
        "procedures": [
            {
                "id": node.id,
                "content": node.content[:300],
                "salience": round(node.metadata.salience, 3),
                "tags": node.metadata.tags,
                "concepts": node.metadata.concepts[:10],
                "agent_id": node.metadata.agent_id,
                "created_at": node.created_at.isoformat(),
            }
            for node in procedures
        ],
    }


@app.get("/procedures/match")
async def match_procedures(
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    concepts: Optional[str] = Query(None, description="Comma-separated concepts"),
):
    """Find procedures matching given tags or concepts."""
    ctx = get_cortex()
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    concept_list = [c.strip() for c in concepts.split(",")] if concepts else None
    procedures = ctx.find_relevant_procedures(tags=tag_list, concepts=concept_list)
    return {
        "count": len(procedures),
        "procedures": [
            {
                "id": node.id,
                "content": node.content[:300],
                "salience": round(node.metadata.salience, 3),
                "tags": node.metadata.tags,
                "concepts": node.metadata.concepts[:10],
                "created_at": node.created_at.isoformat(),
            }
            for node in procedures
        ],
    }


@app.post("/procedures/{procedure_id}/outcome")
async def record_procedure_outcome(procedure_id: str, req: ProcedureOutcomeRequest):
    """Record success or failure of a procedure execution."""
    ctx = get_cortex()
    success = ctx.record_procedure_outcome(procedure_id, req.success)
    if not success:
        raise HTTPException(404, f"Procedure not found: {procedure_id}")
    return {
        "procedure_id": procedure_id,
        "outcome_recorded": True,
        "success": req.success,
    }


# =============================================================================
# Emotions (affective summary / amygdala)
# =============================================================================

@app.get("/emotions/summary")
async def emotional_summary():
    """Get breakdown of memories by emotional valence."""
    ctx = get_cortex()
    summary = ctx.get_emotional_summary()
    return {
        "total": sum(summary.values()),
        "by_valence": summary,
    }


# =============================================================================
# Memory CRUD (get / delete / update)
# NOTE: Placed after /memory/health to avoid path parameter collision
# =============================================================================

@app.get("/memory/{memory_id}")
async def get_memory(memory_id: str, agent_id: Optional[str] = None):
    """Get a single memory by ID."""
    ctx = get_cortex()
    node = ctx.get_memory(memory_id, agent_id=agent_id)
    if not node:
        raise HTTPException(404, f"Memory not found: {memory_id}")
    return {
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
    }


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str, agent_id: Optional[str] = None):
    """Delete a memory from all stores."""
    ctx = get_cortex()
    success = ctx.delete_memory(memory_id, agent_id=agent_id)
    if not success:
        raise HTTPException(404, f"Memory not found: {memory_id}")
    return {"deleted": True, "id": memory_id}


@app.patch("/memory/{memory_id}")
async def update_memory(memory_id: str, req: UpdateMemoryRequest, agent_id: Optional[str] = None):
    """Update a memory's content and/or metadata."""
    ctx = get_cortex()
    visibility = None
    if req.visibility:
        try:
            visibility = Visibility(req.visibility)
        except ValueError:
            raise HTTPException(400, f"Invalid visibility: {req.visibility}")

    updated = ctx.update_memory(
        memory_id=memory_id,
        content=req.content,
        tags=req.tags,
        salience=req.salience,
        visibility=visibility,
        agent_id=agent_id,
    )
    if updated is None:
        raise HTTPException(404, f"Memory not found: {memory_id}")
    return {
        "id": updated.id,
        "content": updated.content,
        "type": updated.metadata.memory_type.value,
        "salience": round(updated.metadata.salience, 3),
        "tags": updated.metadata.tags,
    }


class ShareMemoryRequest(BaseModel):
    visibility: str
    agent_id: Optional[str] = None


@app.post("/memory/{memory_id}/share")
async def share_memory(memory_id: str, req: ShareMemoryRequest):
    """Change a memory's visibility. Only the owner can change visibility."""
    ctx = get_cortex()
    try:
        new_vis = Visibility(req.visibility)
    except ValueError:
        raise HTTPException(400, f"Invalid visibility: {req.visibility}")

    updated = ctx.share_memory(
        memory_id=memory_id,
        new_visibility=new_vis,
        agent_id=req.agent_id,
    )
    if updated is None:
        raise HTTPException(404, f"Not found or not authorized: {memory_id}")
    return {
        "id": updated.id,
        "visibility": updated.metadata.visibility.value,
    }


# =============================================================================
# Dream Engine
# =============================================================================

def _get_dream_engine(ctx: CerebroCortex):
    global _dream_engine
    if _dream_engine is None:
        from cerebro.engines.dream import DreamEngine
        try:
            from cerebro.utils.llm import LLMClient
            llm = LLMClient()
        except Exception:
            llm = None
        _dream_engine = DreamEngine(ctx, llm_client=llm)
    return _dream_engine


@app.post("/dream/run")
async def dream_run():
    """Run a dream consolidation cycle."""
    ctx = get_cortex()
    dream = _get_dream_engine(ctx)

    if dream.is_running:
        raise HTTPException(409, "Dream cycle already in progress")

    report = dream.run_cycle()
    return report.to_dict()


@app.get("/dream/status")
async def dream_status():
    """Get dream engine status and last report."""
    ctx = get_cortex()
    dream = _get_dream_engine(ctx)

    if dream.is_running:
        return {"status": "running"}

    if dream.last_report is None:
        return {"status": "idle", "last_report": None}

    return {
        "status": "idle",
        "last_report": dream.last_report.to_dict(),
    }


# =============================================================================
# Main entry point
# =============================================================================

def main():
    import uvicorn
    logger.info(f"Starting CerebroCortex API on {API_HOST}:{API_PORT}...")
    uvicorn.run(
        "cerebro.interfaces.api_server:app",
        host=API_HOST,
        port=API_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
