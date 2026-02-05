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

    POST /sessions/save   Save session note
    GET  /sessions        Get session notes

    GET  /agents          List agents
    POST /agents          Register agent

    GET  /memory/health          Health report
    GET  /graph/stats            Graph stats
    GET  /graph/neighbors/{id}   Neighbors
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
            "POST /episodes/start", "POST /episodes/{id}/step", "POST /episodes/{id}/end",
            "POST /sessions/save", "GET /sessions",
            "GET /agents", "POST /agents",
            "GET /memory/health", "GET /graph/stats", "GET /graph/neighbors/{id}",
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
