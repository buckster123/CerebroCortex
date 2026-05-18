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

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cerebro.events import event_bus

from cerebro.config import (
    API_HOST,
    API_PORT,
    MCP_SERVER_NAME,
    MCP_SERVER_VERSION,
    WATCH_ENABLED,
    WATCH_DIRS,
    WEB_DIR,
)
from cerebro.cortex import CerebroCortex
from cerebro.models.agent import AgentProfile
from cerebro.settings import (
    apply_settings,
    get_current_settings,
    load_on_startup,
    reset_settings,
)
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

# Serve dashboard static assets (CSS, JS)
if (WEB_DIR / "css").is_dir():
    app.mount("/css", StaticFiles(directory=str(WEB_DIR / "css")), name="css")
if (WEB_DIR / "js").is_dir():
    app.mount("/js", StaticFiles(directory=str(WEB_DIR / "js")), name="js")

# Singleton cortex instance
_cortex: Optional[CerebroCortex] = None
_dream_engine = None


def get_cortex() -> CerebroCortex:
    global _cortex
    if _cortex is None:
        _cortex = CerebroCortex()
        _cortex.initialize()
        logger.info("CerebroCortex initialized")
    return _cortex


@app.on_event("startup")
async def startup_load_settings():
    """Load persisted settings overrides on server start."""
    event_bus.set_loop(asyncio.get_running_loop())
    try:
        load_on_startup()
        logger.info("Settings loaded from disk")
    except Exception as e:
        logger.warning(f"Failed to load settings on startup: {e}")

    # Auto-start file watcher if enabled in settings
    if WATCH_ENABLED and WATCH_DIRS:
        try:
            watcher = _get_watcher()
            watcher.start()
            for d in WATCH_DIRS:
                watcher.add_directory(d)
            logger.info(f"File watcher auto-started with {watcher.watched_count} directorie(s)")
        except Exception as e:
            logger.warning(f"Failed to auto-start file watcher: {e}")


# =============================================================================
# WebSocket
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Real-time event stream for dashboard clients."""
    await ws.accept()
    event_bus.register(ws)
    try:
        # Send welcome event
        await ws.send_text(json.dumps({
            "type": "system:connected",
            "ts": datetime.now().isoformat(),
            "data": {"server_version": MCP_SERVER_VERSION},
        }))
        # Keep alive — just consume any client messages (pings, etc.)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        event_bus.unregister(ws)


# =============================================================================
# Event helpers
# =============================================================================

def _emit_stats_refresh():
    """Emit current node/link/episode counts."""
    try:
        ctx = get_cortex()
        s = ctx.stats()
        event_bus.emit("stats:refresh", {
            "nodes": s["nodes"],
            "links": s["links"],
            "episodes": s["episodes"],
        })
    except Exception:
        pass


def _dream_phase_callback(phase_report, agent_id):
    """Called by DreamEngine after each phase completes."""
    event_bus.emit("dream:phase_complete", {
        "agent_id": agent_id,
        "phase": phase_report.phase.value,
        "success": phase_report.success,
        "metrics": {
            "memories_processed": phase_report.memories_processed,
            "links_created": phase_report.links_created,
            "links_strengthened": phase_report.links_strengthened,
            "memories_pruned": phase_report.memories_pruned,
            "schemas_extracted": phase_report.schemas_extracted,
            "procedures_extracted": phase_report.procedures_extracted,
            "llm_calls": phase_report.llm_calls,
            "duration_seconds": round(phase_report.duration_seconds, 2),
            "notes": phase_report.notes,
        },
    })


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
    offset: int = 0
    memory_types: Optional[list[str]] = None
    agent_id: Optional[str] = None
    min_salience: float = 0.0
    context_ids: Optional[list[str]] = None
    conversation_thread: Optional[str] = None
    explain: bool = False
    include_vision: bool = False

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

class SettingsUpdateRequest(BaseModel):
    llm: Optional[dict] = None
    llm_keys: Optional[dict] = None
    dream: Optional[dict] = None
    scoring: Optional[dict] = None
    advanced: Optional[dict] = None
    watch: Optional[dict] = None


class RenameTagRequest(BaseModel):
    old_tag: str
    new_tag: str
    agent_id: Optional[str] = None


class MergeTagsRequest(BaseModel):
    source_tags: list[str]
    target_tag: str
    agent_id: Optional[str] = None


class BulkDeleteRequest(BaseModel):
    memory_ids: list[str]
    soft: bool = True
    agent_id: Optional[str] = None


class BulkVisibilityRequest(BaseModel):
    memory_ids: list[str]
    visibility: str
    agent_id: Optional[str] = None


class ExportRequest(BaseModel):
    memory_ids: Optional[list[str]] = None
    agent_id: Optional[str] = None
    fmt: str = "json"


class WatchToggleRequest(BaseModel):
    enabled: bool
    dirs: Optional[list[str]] = None


class BootstrapRequest(BaseModel):
    query: str = Field(default="", description="User query for intent analysis")
    mode: Optional[str] = Field(default=None, description="Override mode: minimal, standard, full, custom")
    agent_id: Optional[str] = Field(default=None, description="Agent scope for module recall")
    max_tokens: int = Field(default=4000, description="Maximum token budget for assembled block")


# =============================================================================
# Cognitive Bootstrap Assembler
# =============================================================================

class CognitiveBootstrapAssembler:
    """Assembles a cognitive prompt block from CCBS modules based on query analysis."""

    # Token estimates per module (calibrated on actual content)
    TOKEN_ESTIMATES: dict[str, int] = {
        "soul": 350,
        "module-core": 250,
        "module-cerebro-index": 180,
        "module-cerebro-ops": 320,
        "module-cerebro-session": 280,
        "module-cerebro-intentions": 180,
        "module-cerebro-agents": 180,
        "module-cerebro-meta": 240,
        "module-technical": 320,
        "module-analysis": 280,
        "module-creative": 240,
        "module-research": 240,
        "module-communicate": 220,
    }

    MANDATORY_MODULES = [
        "soul",
        "module-core",
        "module-cerebro-index",
        "module-cerebro-ops",
        "module-cerebro-session",
        "module-cerebro-meta",
    ]

    # Keyword -> module name mapping for auto-detection
    KEYWORD_MAP: dict[str, list[str]] = {
        "technical": ["module-technical"],
        "analysis": ["module-analysis"],
        "creative": ["module-creative"],
        "research": ["module-research"],
        "communicate": ["module-communicate"],
    }

    # Flat keyword -> module mapping for quick lookup
    _KEYWORD_TRIGGERS: dict[str, list[str]] = {
        # Technical
        "python": ["module-technical"], "javascript": ["module-technical"], "typescript": ["module-technical"],
        "rust": ["module-technical"], "go": ["module-technical"], "java": ["module-technical"],
        "code": ["module-technical"], "bug": ["module-technical", "module-analysis"],
        "fix": ["module-technical", "module-analysis"], "error": ["module-technical", "module-analysis"],
        "debug": ["module-technical", "module-analysis"], "architecture": ["module-technical"],
        "api": ["module-technical"], "server": ["module-technical"], "database": ["module-technical"],
        "docker": ["module-technical"], "git": ["module-technical"], "build": ["module-technical"],
        "test": ["module-technical"], "pytest": ["module-technical"], "deploy": ["module-technical"],
        "compile": ["module-technical"], "endpoint": ["module-technical"], "schema": ["module-technical"],
        # Analysis
        "why": ["module-analysis"], "evaluate": ["module-analysis"], "compare": ["module-analysis"],
        "versus": ["module-analysis"], "trade-off": ["module-analysis"], "measure": ["module-analysis"],
        "benchmark": ["module-analysis"], "profile": ["module-analysis"], "performance": ["module-analysis"],
        "latency": ["module-analysis"], "throughput": ["module-analysis"], "root cause": ["module-analysis"],
        "investigate": ["module-analysis"], "diagnose": ["module-analysis"], "reproduce": ["module-analysis"],
        # Creative
        "design": ["module-creative"], "create": ["module-creative"], "ideate": ["module-creative"],
        "brainstorm": ["module-creative"], "concept": ["module-creative"], "theme": ["module-creative"],
        "palette": ["module-creative"], "layout": ["module-creative"], "logo": ["module-creative"],
        "brand": ["module-creative"], "ui": ["module-creative"], "ux": ["module-creative"],
        "visual": ["module-creative"], "animation": ["module-creative"], "style": ["module-creative"],
        # Research
        "paper": ["module-research"], "arxiv": ["module-research"], "study": ["module-research"],
        "research": ["module-research"], "state of": ["module-research"], "survey": ["module-research"],
        "review": ["module-research"], "literature": ["module-research"], "dataset": ["module-research"],
        "benchmark results": ["module-research"], "evaluation": ["module-research", "module-analysis"],
        # Communicate
        "explain": ["module-communicate"], "teach": ["module-communicate"], "how to": ["module-communicate"],
        "what is": ["module-communicate"], "document": ["module-communicate"], "write": ["module-communicate"],
        "describe": ["module-communicate"], "summarize": ["module-communicate"], "brief": ["module-communicate"],
        "report": ["module-communicate"], "presentation": ["module-communicate"], "diagram": ["module-communicate"],
        # Intentions
        "todo": ["module-cerebro-intentions"], "plan": ["module-cerebro-intentions"],
        "remember to": ["module-cerebro-intentions"], "don't forget": ["module-cerebro-intentions"],
        "later": ["module-cerebro-intentions"], "next": ["module-cerebro-intentions"],
        # Agents
        "agent": ["module-cerebro-agents"], "message": ["module-cerebro-agents"],
        "hailo": ["module-cerebro-agents"], "apex": ["module-cerebro-agents"],
        "send to": ["module-cerebro-agents"], "inbox": ["module-cerebro-agents"],
    }

    # Manual trigger -> modules mapping
    MANUAL_TRIGGERS: dict[str, list[str]] = {
        "full load": [],  # special: all modules
        "max brain": [],
        "all in": [],
        "solo core": ["solo"],
        "minimal": ["solo"],
        "debug mode": ["module-technical", "module-analysis"],
        "creative mode": ["module-creative"],
        "research mode": ["module-research", "module-analysis"],
        "cerebro mode": ["module-cerebro-intentions", "module-cerebro-agents"],
        "teach me": ["module-communicate"],
        "explain": ["module-communicate"],
    }

    def __init__(self, cortex: CerebroCortex):
        self.cortex = cortex

    def assemble(
        self,
        query: str,
        mode: Optional[str] = None,
        agent_id: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> dict:
        """Analyze query and assemble the cognitive block."""
        query_lower = query.lower()

        # Step 1: Detect manual triggers
        trigger_modules, detected_trigger = self._detect_triggers(query_lower)

        # Step 2: Auto-detect from keywords (if no manual trigger or mode is standard+)
        keyword_modules = set()
        if not detected_trigger or mode in (None, "standard", "full"):
            keyword_modules = self._detect_keywords(query_lower)

        # Step 3: Determine mode and module set
        if detected_trigger == "solo":
            mode = mode or "minimal"
            module_names = ["soul", "module-core"]
        elif detected_trigger in ("full load", "max brain", "all in"):
            mode = "full"
            module_names = list(self.TOKEN_ESTIMATES.keys())
        else:
            mode = mode or "standard"
            module_names = list(self.MANDATORY_MODULES)
            if trigger_modules:
                module_names.extend(trigger_modules)
            module_names.extend(keyword_modules)

        module_names = list(dict.fromkeys(module_names))  # preserve order, dedup

        # Step 4: Token budget enforcement
        module_names = self._enforce_budget(module_names, max_tokens, mode)

        # Step 5: Load module contents from Cerebro
        loaded = []
        missing = []
        for name in module_names:
            content = self._load_module_content(name, agent_id=agent_id)
            if content:
                loaded.append({"name": name, "content": content, "tokens": self.TOKEN_ESTIMATES.get(name, 300)})
            else:
                missing.append(name)

        total_tokens = sum(m["tokens"] for m in loaded)

        return {
            "mode": mode,
            "trigger": detected_trigger,
            "modules_loaded": [m["name"] for m in loaded],
            "modules_missing": missing,
            "total_tokens": total_tokens,
            "max_tokens": max_tokens,
            "assembled_block": "\n\n---\n\n".join(m["content"] for m in loaded),
            "query": query,
        }

    def _detect_triggers(self, query: str) -> tuple[list[str], Optional[str]]:
        for trigger, modules in self.MANUAL_TRIGGERS.items():
            if trigger in query:
                if trigger in ("full load", "max brain", "all in"):
                    return [], trigger
                if trigger in ("solo core", "minimal"):
                    return [], "solo"
                return modules, trigger
        return [], None

    def _detect_keywords(self, query: str) -> set[str]:
        found = set()
        for keyword, modules in self._KEYWORD_TRIGGERS.items():
            if keyword in query:
                found.update(modules)
        return found

    def _enforce_budget(self, module_names: list[str], max_tokens: int, mode: str) -> list[str]:
        if mode == "minimal":
            max_tokens = min(max_tokens, 1000)
        elif mode == "standard":
            max_tokens = min(max_tokens, 2000)
        elif mode == "full":
            max_tokens = min(max_tokens, 4500)

        total = 0
        result = []
        for name in module_names:
            cost = self.TOKEN_ESTIMATES.get(name, 300)
            if total + cost > max_tokens:
                break
            total += cost
            result.append(name)
        return result

    def _load_module_content(self, name: str, agent_id: Optional[str] = None) -> Optional[str]:
        """Load a module's content from Cerebro by header search."""
        try:
            # Search by exact module header for reliable retrieval
            header_match = name.replace("module-", "")
            results = self.cortex.recall(
                query=f"# Module: {header_match}",
                top_k=20,
                agent_id=agent_id,
                memory_types=[MemoryType.PROCEDURAL, MemoryType.SEMANTIC],
            )
            # Find exact match by header
            for node, score in results:
                if f"# Module: {header_match}" in node.content:
                    return node.content
            return None
        except Exception as exc:
            logger.warning("Failed to load module %s: %s", name, exc)
            return None


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
async def stats(agent_id: Optional[str] = None):
    """Full system statistics, optionally scoped to an agent."""
    ctx = get_cortex()
    return ctx.stats(agent_id=agent_id)


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

    result = {
        "stored": True,
        "id": node.id,
        "type": node.metadata.memory_type.value,
        "layer": node.metadata.layer.value,
        "salience": round(node.metadata.salience, 3),
        "valence": node.metadata.valence.value if hasattr(node.metadata.valence, "value") else str(node.metadata.valence),
        "concepts": node.metadata.concepts[:10],
        "links": node.link_count,
    }
    event_bus.emit("memory:stored", {
        "id": node.id,
        "type": node.metadata.memory_type.value,
        "layer": node.metadata.layer.value,
        "salience": round(node.metadata.salience, 3),
    })
    _emit_stats_refresh()
    return result


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
        offset=req.offset,
        memory_types=memory_types,
        agent_id=req.agent_id,
        min_salience=req.min_salience,
        context_ids=req.context_ids,
        conversation_thread=req.conversation_thread,
        explain=req.explain,
        include_vision=req.include_vision,
    )

    def _fmt_result(entry):
        if len(entry) == 3:
            node, score, breakdown = entry
            base = {
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
                "score_breakdown": breakdown,
            }
        else:
            node, score = entry
            base = {
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
        return base

    return {
        "query": req.query,
        "count": len(results),
        "results": [_fmt_result(r) for r in results],
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

    result = {
        "link_id": link_id,
        "source_id": req.source_id,
        "target_id": req.target_id,
        "link_type": req.link_type,
        "weight": req.weight,
    }
    event_bus.emit("link:created", {
        "source_id": req.source_id,
        "target_id": req.target_id,
        "link_type": req.link_type,
    })
    _emit_stats_refresh()
    return result


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
async def get_episode(episode_id: str, agent_id: Optional[str] = None):
    """Get a single episode by ID."""
    ctx = get_cortex()
    episode = ctx.get_episode(episode_id, agent_id=agent_id)
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
async def get_episode_memories(episode_id: str, agent_id: Optional[str] = None):
    """Get all memories in an episode, ordered by position."""
    ctx = get_cortex()
    memories = ctx.get_episode_memories(episode_id, agent_id=agent_id)
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
    agent_id: Optional[str] = None,
):
    """Get graph data (nodes + links) for visualization."""
    ctx = get_cortex()

    # Get nodes from SQLite, ordered by salience descending
    if agent_id:
        rows = ctx.graph.conn.execute(
            "SELECT id, content, memory_type, layer, salience, valence, "
            "arousal, tags_json, concepts_json, agent_id, access_count, "
            "created_at FROM memory_nodes "
            "WHERE salience >= ? "
            "AND (visibility='shared' OR (visibility='private' AND agent_id=?) "
            "OR (visibility='thread' AND agent_id=?)) "
            "ORDER BY salience DESC LIMIT ?",
            (min_salience, agent_id, agent_id, limit),
        ).fetchall()
    else:
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
            for nid, w, lt, *_ in neighbors
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
# Activation & Decay (ACT-R + FSRS visualization)
# =============================================================================

@app.get("/activation/heatmap")
async def activation_heatmap(
    limit: int = Query(200, ge=1, le=1000),
    agent_id: Optional[str] = None,
):
    """Get activation data for all memories, suitable for scatter-plot visualization.

    Returns {memory_id, age_hours, activation, retrievability, layer, salience}[]
    """
    import math
    from cerebro.activation.strength import base_level_activation, retrievability

    ctx = get_cortex()
    query = "SELECT id FROM memory_nodes WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT ?"
    rows = ctx._graph.conn.execute(query, (limit,)).fetchall()

    now = time.time()
    points = []
    for row in rows:
        node = ctx._graph.get_node(row["id"])
        if not node:
            continue
        if not ctx._can_access(node, agent_id):
            continue

        strength = node.strength
        last_access = strength.access_timestamps[-1] if strength.access_timestamps else (
            node.created_at.timestamp() if hasattr(node.created_at, 'timestamp') else now
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

    return {"count": len(points), "points": points}


@app.get("/activation/at-risk")
async def activation_at_risk(
    hours: int = Query(24, ge=1, le=720),
    layer: Optional[str] = None,
    agent_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """Get memories that are fading (low activation / low retrievability / old).

    Args:
        hours: Hours since last access to be considered "at risk"
        layer: Filter by layer (sensory, working, long_term)
        agent_id: Filter by agent visibility
        limit: Max results
    """
    ctx = get_cortex()
    at_risk = ctx.get_at_risk_memories(
        hours=hours, layer=layer, agent_id=agent_id, limit=limit,
    )
    return {
        "count": len(at_risk),
        "threshold_hours": hours,
        "memories": [
            {
                "id": node.id,
                "content_preview": node.content[:200],
                "activation": round(activation, 3),
                "retrievability": round(r, 3),
                "hours_since_access": round(hours_since, 1),
                "layer": node.metadata.layer.value,
                "salience": node.metadata.salience,
            }
            for node, activation, r, hours_since in at_risk
        ],
    }


@app.get("/activation/curve/{memory_id}")
async def activation_curve(memory_id: str, days: int = Query(30, ge=1, le=90)):
    """Get the projected ACT-R decay curve for a specific memory."""
    from cerebro.activation.strength import compute_activation_curve

    ctx = get_cortex()
    node = ctx.get_memory(memory_id)
    if not node:
        raise HTTPException(404, f"Memory not found: {memory_id}")

    curve = compute_activation_curve(node.strength, days=days)
    return {
        "memory_id": memory_id,
        "days": days,
        "curve": curve,
    }


# =============================================================================
# Audit logging
# =============================================================================

class AuditQueryRequest(BaseModel):
    event_type: Optional[str] = None
    actor: Optional[str] = None
    target: Optional[str] = None
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)


@app.post("/audit/query")
async def query_audit(req: AuditQueryRequest):
    """Query the audit log with optional filters."""
    ctx = get_cortex()
    entries = ctx._graph.query_audit(
        event_type=req.event_type,
        actor=req.actor,
        target=req.target,
        limit=req.limit,
        offset=req.offset,
    )
    total = ctx._graph.count_audit(event_type=req.event_type, actor=req.actor)
    return {
        "count": len(entries),
        "total": total,
        "offset": req.offset,
        "limit": req.limit,
        "entries": entries,
    }


@app.get("/audit/summary")
async def audit_summary():
    """Get a summary of audit events by type."""
    ctx = get_cortex()
    rows = ctx._graph.conn.execute(
        "SELECT event_type, COUNT(*) as c FROM audit_log GROUP BY event_type ORDER BY c DESC"
    ).fetchall()
    return {
        "total_events": sum(r["c"] for r in rows),
        "by_type": [{"event_type": r["event_type"], "count": r["c"]} for r in rows],
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
    event_bus.emit("memory:deleted", {"id": memory_id})
    _emit_stats_refresh()
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
    result = {
        "id": updated.id,
        "content": updated.content,
        "type": updated.metadata.memory_type.value,
        "salience": round(updated.metadata.salience, 3),
        "tags": updated.metadata.tags,
    }
    event_bus.emit("memory:updated", {
        "id": updated.id,
        "type": updated.metadata.memory_type.value,
        "salience": round(updated.metadata.salience, 3),
    })
    return result


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
        _dream_engine = DreamEngine(
            ctx, llm_client=llm,
            on_phase_complete=_dream_phase_callback,
        )
    return _dream_engine


def _run_dream_in_thread(dream):
    """Run dream cycle in a background thread, emitting events on completion."""
    try:
        event_bus.emit("dream:started", {})
        reports = dream.run_all_agents_cycle()
        event_bus.emit("dream:complete", {
            "reports": [r.to_dict() for r in reports],
        })
        _emit_stats_refresh()
    except Exception as e:
        logger.error(f"Dream thread failed: {e}", exc_info=True)
        event_bus.emit("dream:error", {"error": str(e)})


@app.post("/dream/run")
async def dream_run():
    """Run a dream consolidation cycle (non-blocking)."""
    ctx = get_cortex()
    dream = _get_dream_engine(ctx)

    if dream.is_running:
        raise HTTPException(409, "Dream cycle already in progress")

    threading.Thread(
        target=_run_dream_in_thread,
        args=(dream,),
        daemon=True,
    ).start()

    return {"status": "started"}


@app.get("/dream/status")
async def dream_status(agent_id: Optional[str] = None):
    """Get dream engine status and last report(s).

    If agent_id is provided, returns the report for that specific agent.
    Otherwise returns all reports from the last cycle.
    """
    ctx = get_cortex()
    dream = _get_dream_engine(ctx)

    if dream.is_running:
        return {"status": "running"}

    reports = dream.last_reports
    if agent_id and reports:
        # Find report for specific agent
        match = next((r for r in reports if r.agent_id == agent_id), None)
        if match:
            return {"status": "idle", "last_report": match.to_dict()}
        return {"status": "idle", "last_report": None}

    if reports:
        return {
            "status": "idle",
            "reports": [r.to_dict() for r in reports],
        }

    # Fallback to single last_report (for single-agent cycles)
    if dream.last_report is None:
        return {"status": "idle", "last_report": None}

    return {
        "status": "idle",
        "last_report": dream.last_report.to_dict(),
    }


# =============================================================================
# Settings
# =============================================================================

@app.get("/settings")
async def get_settings(dev: bool = False):
    """Get current settings (masks API keys)."""
    return get_current_settings(include_dev=dev)


@app.put("/settings")
async def update_settings(req: SettingsUpdateRequest):
    """Partial update of settings. Persists to settings.json and hot-reloads."""
    global _dream_engine
    updates = {}
    for section in ("llm", "llm_keys", "dream", "scoring", "advanced", "watch"):
        val = getattr(req, section, None)
        if val is not None:
            updates[section] = val

    if not updates:
        raise HTTPException(400, "No settings provided")

    applied = apply_settings(updates)

    # Reset dream engine if LLM config changed so it picks up new settings
    if any(k.startswith("llm.") for k in applied):
        _dream_engine = None

    return {"applied": applied, "count": len(applied)}


@app.post("/settings/reset")
async def reset_all_settings():
    """Reset all settings to config.py defaults."""
    global _dream_engine
    reset_settings()
    _dream_engine = None
    return {"reset": True}


# =============================================================================
# Ingestion (file upload)
# =============================================================================

@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    tags: Optional[str] = None,
    agent_id: Optional[str] = None,
    dedup: bool = True,
):
    """Upload and ingest a single file.

    Args:
        file: The file to upload.
        tags: Comma-separated tags (e.g. "project,research").
        agent_id: Agent ID for attribution.
        dedup: If False, bypass near-duplicate detection (default True).
    """
    import tempfile

    ctx = get_cortex()
    tag_list = [t.strip() for t in tags.split(",")] if tags else ["uploaded"]
    agent = agent_id or os.environ.get("CEREBRO_AGENT_ID", "CLAUDE")

    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from cerebro.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(ctx)
        report = pipeline.ingest_file(tmp_path, tags=tag_list, agent_id=agent)
        if report.errors:
            raise HTTPException(400, f"Ingestion errors: {report.errors}")
        _emit_stats_refresh()
        return {
            "file": file.filename,
            "memories_created": report.memories_imported,
            "memories_skipped": report.memories_skipped,
            "duration_seconds": round(report.duration_seconds, 3),
        }
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# =============================================================================
# Near-duplicate detection
# =============================================================================

class NearDuplicateCheckRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Content to check for near-duplicates")
    threshold: float = Field(0.95, ge=0.0, le=1.0)
    agent_id: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)


@app.post("/near-duplicates/check")
async def check_near_duplicates(req: NearDuplicateCheckRequest):
    """Check if content would be a near-duplicate of existing memories.

    Returns existing memories with similarity scores above the threshold.
    This is useful for client-side deduplication previews before ingestion.
    """
    ctx = get_cortex()
    matches = ctx.find_near_duplicates(
        content=req.content,
        threshold=req.threshold,
        agent_id=req.agent_id,
        top_k=req.top_k,
    )
    return {
        "threshold": req.threshold,
        "matches_found": len(matches),
        "matches": [
            {
                "id": m["id"],
                "similarity": m.get("similarity"),
                "content_preview": m.get("content", "")[:200],
                "collection": m.get("collection"),
            }
            for m in matches
        ],
    }


# =============================================================================
# Trash (soft-deleted memories)
# =============================================================================

@app.get("/trash")
async def list_trash(agent_id: Optional[str] = None, limit: int = Query(50, ge=1, le=200)):
    """List soft-deleted memories."""
    ctx = get_cortex()
    nodes = ctx.list_deleted(agent_id=agent_id, limit=limit)
    return {
        "count": len(nodes),
        "memories": [
            {
                "id": n.id,
                "content": n.content,
                "type": n.metadata.memory_type.value,
                "deleted_at": n.metadata.deleted_at if hasattr(n.metadata, 'deleted_at') else None,
                "tags": n.metadata.tags,
            }
            for n in nodes
        ],
    }


@app.post("/trash/{memory_id}/restore")
async def restore_trash(memory_id: str, agent_id: Optional[str] = None):
    """Restore a soft-deleted memory."""
    ctx = get_cortex()
    success = ctx.restore_memory(memory_id, agent_id=agent_id)
    if not success:
        raise HTTPException(404, f"Memory not found in trash: {memory_id}")
    _emit_stats_refresh()
    return {"restored": True, "id": memory_id}


@app.delete("/trash/{memory_id}")
async def purge_trash(memory_id: str, agent_id: Optional[str] = None):
    """Permanently delete a memory from trash."""
    ctx = get_cortex()
    success = ctx.purge_memory(memory_id, agent_id=agent_id)
    if not success:
        raise HTTPException(404, f"Memory not found: {memory_id}")
    return {"purged": True, "id": memory_id}


@app.post("/trash/purge-all")
async def purge_all_trash(older_than_days: int = Query(0, ge=0), agent_id: Optional[str] = None):
    """Purge all soft-deleted memories. Use older_than_days to only purge stale items."""
    ctx = get_cortex()
    count = ctx.purge_all_deleted(older_than_days=older_than_days, agent_id=agent_id)
    return {"purged": count}


# =============================================================================
# Memory Versions
# =============================================================================

@app.get("/memory/{memory_id}/versions")
async def get_versions(memory_id: str, limit: int = Query(10, ge=1, le=50)):
    """Get version history for a memory."""
    ctx = get_cortex()
    versions = ctx.get_memory_versions(memory_id, limit=limit)
    return {"memory_id": memory_id, "count": len(versions), "versions": versions}


@app.post("/memory/{memory_id}/versions/{version_id}/restore")
async def restore_version(memory_id: str, version_id: int, agent_id: Optional[str] = None):
    """Restore a memory to a previous version."""
    ctx = get_cortex()
    node = ctx.restore_version(version_id, agent_id=agent_id)
    if node is None:
        raise HTTPException(404, f"Version not found: {version_id}")
    return {"restored": True, "memory_id": node.id, "version_id": version_id}


# =============================================================================
# Tags
# =============================================================================

@app.get("/tags")
async def list_tags(agent_id: Optional[str] = None):
    """List all tags with usage counts."""
    ctx = get_cortex()
    tags = ctx.tag_manager.list_tags(agent_id=agent_id)
    return {"tags": [{"name": k, "count": v} for k, v in sorted(tags.items())]}


@app.post("/tags/rename")
async def rename_tag(req: RenameTagRequest):
    """Rename a tag across all memories."""
    ctx = get_cortex()
    count = ctx.tag_manager.rename_tag(req.old_tag, req.new_tag, agent_id=req.agent_id)
    return {"renamed": count, "old": req.old_tag, "new": req.new_tag}


@app.post("/tags/merge")
async def merge_tags(req: MergeTagsRequest):
    """Merge multiple tags into one."""
    ctx = get_cortex()
    count = ctx.tag_manager.merge_tags(req.source_tags, req.target_tag, agent_id=req.agent_id)
    return {"merged": count, "sources": req.source_tags, "target": req.target_tag}


@app.delete("/tags/{tag}")
async def delete_tag(tag: str, agent_id: Optional[str] = None):
    """Remove a tag from all memories."""
    ctx = get_cortex()
    ctx.tag_manager.delete_tag(tag, agent_id=agent_id)
    return {"deleted": True, "tag": tag}


# =============================================================================
# Threads
# =============================================================================

@app.get("/threads")
async def list_threads(agent_id: Optional[str] = None, limit: int = Query(50, ge=1, le=200)):
    """List conversation threads with memory counts."""
    ctx = get_cortex()
    threads = ctx.list_threads(agent_id=agent_id, limit=limit)
    return {"count": len(threads), "threads": threads}


@app.get("/threads/{thread_id}/memories")
async def get_thread_memories(thread_id: str, limit: int = Query(100, ge=1, le=500)):
    """Get all memories in a conversation thread."""
    ctx = get_cortex()
    results = ctx.get_thread_memories(thread_id, limit=limit)
    return {
        "thread_id": thread_id,
        "count": len(results),
        "memories": [
            {
                "id": node.id,
                "content": node.content,
                "type": node.metadata.memory_type.value,
                "created_at": node.created_at.isoformat(),
            }
            for node, _score in results
        ],
    }


@app.delete("/threads/{thread_id}")
async def prune_thread(thread_id: str, agent_id: Optional[str] = None):
    """Soft-delete all memories in a thread."""
    ctx = get_cortex()
    count = ctx.prune_thread(thread_id, agent_id=agent_id)
    _emit_stats_refresh()
    return {"pruned": count, "thread_id": thread_id}


# =============================================================================
# Bulk Operations
# =============================================================================

@app.post("/bulk/delete")
async def bulk_delete(req: BulkDeleteRequest):
    """Soft-delete multiple memories at once."""
    ctx = get_cortex()
    deleted = ctx.bulk_delete(req.memory_ids, soft=req.soft, agent_id=req.agent_id)
    _emit_stats_refresh()
    return {"deleted": len(deleted), "ids": deleted}


@app.post("/bulk/visibility")
async def bulk_visibility(req: BulkVisibilityRequest):
    """Change visibility for multiple memories at once."""
    ctx = get_cortex()
    try:
        vis = Visibility(req.visibility)
    except ValueError:
        raise HTTPException(400, f"Invalid visibility: {req.visibility}")
    updated = ctx.bulk_update_visibility(req.memory_ids, visibility=vis, agent_id=req.agent_id)
    return {"updated": len(updated), "ids": updated}


@app.post("/export")
async def export_memories(req: ExportRequest):
    """Export memories to JSON or Markdown."""
    ctx = get_cortex()
    export_path = ctx.export_memories(
        memory_ids=req.memory_ids,
        agent_id=req.agent_id,
        fmt=req.fmt,
    )
    return {"path": str(export_path), "format": req.fmt}


# =============================================================================
# File Watcher
# =============================================================================

_watcher = None


def _get_watcher():
    global _watcher
    if _watcher is None:
        from cerebro.watch import FileWatcher
        _watcher = FileWatcher(
            get_cortex(),
            agent_id=os.environ.get("CEREBRO_AGENT_ID", "CLAUDE"),
        )
    return _watcher




@app.get("/browse")
async def browse_directories(path: Optional[str] = None):
    """Browse directories for folder picker. Scoped to home directory for safety.

    Returns: {current: str, parent: str|null, entries: [{name, type, path}]}
    """
    home = Path.home().resolve()
    if path:
        target = Path(path).expanduser().resolve()
    else:
        target = home

    # Safety: prevent escaping home directory
    try:
        target.relative_to(home)
    except ValueError:
        target = home

    if not target.exists():
        return {"current": str(target), "parent": None, "entries": [], "error": "Path not found"}
    if not target.is_dir():
        target = target.parent

    parent = str(target.parent) if target != home else None

    entries = []
    try:
        for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name.startswith("."):
                continue
            entries.append({
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
                "path": str(entry),
            })
    except PermissionError:
        pass

    return {"current": str(target), "parent": parent, "entries": entries}

@app.get("/watch/status")
async def watch_status():
    """Get file watcher status."""
    global _watcher
    return {
        "enabled": WATCH_ENABLED,
        "running": _watcher.is_running() if _watcher else False,
        "watched_directories": WATCH_DIRS,
        "watched_count": _watcher.watched_count if _watcher else 0,
    }


@app.post("/watch/toggle")
async def watch_toggle(req: WatchToggleRequest):
    """Enable or disable the file watcher."""
    global _watcher
    import cerebro.config as cfg

    cfg.WATCH_ENABLED = req.enabled
    apply_settings({"watch": {"enabled": req.enabled}})

    watcher = _get_watcher()
    if req.enabled:
        if not watcher.is_running():
            watcher.start()
        dirs = req.dirs or WATCH_DIRS or []
        # Auto-create default watch dir if none configured
        if not dirs:
            default_dir = DEFAULT_WATCH_DIR
            default_dir.mkdir(parents=True, exist_ok=True)
            dirs = [str(default_dir)]
            apply_settings({"watch": {"dirs": dirs}})
        for d in dirs:
            watcher.add_directory(d)
    else:
        watcher.stop()
        _watcher = None

    return {"enabled": req.enabled}


# =============================================================================
# Cognitive Bootstrap Endpoint
# =============================================================================

@app.post("/bootstrap")
async def bootstrap(req: BootstrapRequest):
    """Assemble a cognitive prompt block from CCBS modules based on query analysis.

    Returns an assembled block of cognitive modules tailored to the session intent,
    with manual trigger support, keyword auto-detection, and token budget enforcement.
    """
    ctx = get_cortex()
    assembler = CognitiveBootstrapAssembler(ctx)
    return assembler.assemble(
        query=req.query,
        mode=req.mode,
        agent_id=req.agent_id,
        max_tokens=req.max_tokens,
    )


# =============================================================================
# Main entry point
# =============================================================================

def main():
    import uvicorn
    from cerebro._init_dirs import ensure_data_dirs
    ensure_data_dirs()
    logger.info(f"Starting CerebroCortex API on {API_HOST}:{API_PORT}...")
    uvicorn.run(
        "cerebro.interfaces.api_server:app",
        host=API_HOST,
        port=API_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
