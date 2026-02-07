"""CerebroCortex CLI.

Click-based command line interface for CerebroCortex.

Usage:
    cerebro stats
    cerebro remember "Python uses indentation for blocks"
    cerebro recall "programming languages"
    cerebro associate <src_id> <tgt_id> semantic
    cerebro episode start "Debug session"
    cerebro session save "Built the REST API"
    cerebro session recall
    cerebro agents list
    cerebro health
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


import click

from cerebro.cortex import CerebroCortex
from cerebro.models.agent import AgentProfile
from cerebro.types import EmotionalValence, LinkType, MemoryType, Visibility

_cortex: Optional[CerebroCortex] = None


def get_cortex() -> CerebroCortex:
    global _cortex
    if _cortex is None:
        _cortex = CerebroCortex()
        _cortex.initialize()
    return _cortex


def _json_out(data):
    """Print data as JSON."""
    click.echo(json.dumps(data, indent=2, default=str))


# =============================================================================
# Root group
# =============================================================================

@click.group()
def cli():
    """CerebroCortex - Brain-analogous AI memory system."""
    pass


# =============================================================================
# Stats
# =============================================================================

@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def stats(as_json):
    """Show system statistics."""
    ctx = get_cortex()
    s = ctx.stats()

    if as_json:
        _json_out(s)
        return

    click.echo("CerebroCortex Statistics")
    click.echo("=" * 40)
    click.echo(f"  Memories:    {s['nodes']}")
    click.echo(f"  Links:       {s['links']}")
    click.echo(f"  Episodes:    {s['episodes']}")
    click.echo(f"  Schemas:     {s['schemas']}")
    click.echo(f"  igraph:      {s['igraph_vertices']} vertices, {s['igraph_edges']} edges")

    if s.get("memory_types"):
        click.echo("\nBy Type:")
        for t, c in s["memory_types"].items():
            if c > 0:
                click.echo(f"  {t:15s} {c}")

    if s.get("layers"):
        click.echo("\nBy Layer:")
        for l, c in s["layers"].items():
            if c > 0:
                click.echo(f"  {l:15s} {c}")

    if s.get("link_types"):
        click.echo("\nLink Types:")
        for lt, c in s["link_types"].items():
            if c > 0:
                click.echo(f"  {lt:15s} {c}")


# =============================================================================
# Remember
# =============================================================================

@cli.command()
@click.argument("content")
@click.option("--type", "memory_type", type=click.Choice(
    ["episodic", "semantic", "procedural", "affective", "prospective", "schematic"],
), help="Memory type")
@click.option("--tags", multiple=True, help="Tags (repeatable)")
@click.option("--salience", type=float, help="Importance 0-1")
@click.option("--agent", default="CLAUDE", help="Agent ID")
@click.option("--visibility", type=click.Choice(["shared", "private", "thread"]), default="shared")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def remember(content, memory_type, tags, salience, agent, visibility, as_json):
    """Store a memory."""
    ctx = get_cortex()

    mt = MemoryType(memory_type) if memory_type else None
    vis = Visibility(visibility)

    node = ctx.remember(
        content=content,
        memory_type=mt,
        tags=list(tags) if tags else None,
        salience=salience,
        agent_id=agent,
        visibility=vis,
    )

    if node is None:
        if as_json:
            _json_out({"stored": False, "reason": "gated_out"})
        else:
            click.echo("Memory gated out (duplicate or too short).")
        return

    if as_json:
        _json_out({
            "stored": True,
            "id": node.id,
            "type": node.metadata.memory_type.value,
            "layer": node.metadata.layer.value,
            "salience": round(node.metadata.salience, 3),
            "concepts": node.metadata.concepts[:5],
            "links": node.link_count,
        })
    else:
        click.echo(f"Stored: {node.id}")
        click.echo(f"  Type:     {node.metadata.memory_type.value}")
        click.echo(f"  Layer:    {node.metadata.layer.value}")
        click.echo(f"  Salience: {node.metadata.salience:.2f}")
        if node.metadata.concepts:
            click.echo(f"  Concepts: {', '.join(node.metadata.concepts[:5])}")
        click.echo(f"  Links:    {node.link_count}")


# =============================================================================
# Recall
# =============================================================================

@cli.command()
@click.argument("query")
@click.option("-n", "--results", default=10, help="Max results")
@click.option("--type", "memory_type", type=click.Choice(
    ["episodic", "semantic", "procedural", "affective", "prospective", "schematic"],
), help="Filter by type")
@click.option("--agent", help="Filter by agent")
@click.option("--min-salience", type=float, default=0.0, help="Minimum salience")
@click.option("--thread", help="Conversation thread ID for THREAD visibility")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def recall(query, results, memory_type, agent, min_salience, thread, as_json):
    """Search and retrieve memories."""
    ctx = get_cortex()

    memory_types = [MemoryType(memory_type)] if memory_type else None

    found = ctx.recall(
        query=query,
        top_k=results,
        memory_types=memory_types,
        agent_id=agent,
        min_salience=min_salience,
        conversation_thread=thread,
    )

    if as_json:
        _json_out({
            "query": query,
            "count": len(found),
            "results": [
                {
                    "id": node.id,
                    "content": node.content,
                    "type": node.metadata.memory_type.value,
                    "salience": round(node.metadata.salience, 3),
                    "score": round(score, 4),
                    "tags": node.metadata.tags,
                }
                for node, score in found
            ],
        })
        return

    if not found:
        click.echo("No memories found.")
        return

    click.echo(f"Found {len(found)} memories:\n")
    for i, (node, score) in enumerate(found, 1):
        preview = node.content[:120] + "..." if len(node.content) > 120 else node.content
        click.echo(f"  {i}. [{node.metadata.memory_type.value}] score={score:.3f} sal={node.metadata.salience:.2f}")
        click.echo(f"     {preview}")
        if node.metadata.tags:
            click.echo(f"     tags: {', '.join(node.metadata.tags)}")
        click.echo(f"     id: {node.id}")
        click.echo()


# =============================================================================
# Get / Delete / Update memory
# =============================================================================

@cli.command("get")
@click.argument("memory_id")
@click.option("--agent", help="Agent ID for access check")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def get_memory(memory_id, agent, as_json):
    """Get a memory by ID."""
    ctx = get_cortex()
    node = ctx.get_memory(memory_id, agent_id=agent)
    if not node:
        click.echo(f"Memory not found: {memory_id}", err=True)
        sys.exit(1)

    if as_json:
        _json_out({
            "id": node.id,
            "content": node.content,
            "type": node.metadata.memory_type.value,
            "layer": node.metadata.layer.value,
            "salience": round(node.metadata.salience, 3),
            "tags": node.metadata.tags,
            "concepts": node.metadata.concepts,
            "agent_id": node.metadata.agent_id,
            "created_at": node.created_at.isoformat(),
        })
        return

    click.echo(f"ID: {node.id}")
    click.echo(f"  Type:     {node.metadata.memory_type.value}")
    click.echo(f"  Layer:    {node.metadata.layer.value}")
    click.echo(f"  Salience: {node.metadata.salience:.2f}")
    click.echo(f"  Tags:     {', '.join(node.metadata.tags) if node.metadata.tags else 'none'}")
    click.echo(f"  Concepts: {', '.join(node.metadata.concepts[:5]) if node.metadata.concepts else 'none'}")
    click.echo(f"  Agent:    {node.metadata.agent_id}")
    click.echo(f"  Created:  {node.created_at.strftime('%Y-%m-%d %H:%M')}")
    click.echo(f"\n{node.content}")


@cli.command("delete")
@click.argument("memory_id")
@click.option("--agent", help="Agent ID for access check")
@click.option("--force", is_flag=True, help="Skip confirmation")
def delete_memory(memory_id, agent, force):
    """Delete a memory from all stores."""
    ctx = get_cortex()

    if not force:
        node = ctx.get_memory(memory_id, agent_id=agent)
        if not node:
            click.echo(f"Memory not found: {memory_id}", err=True)
            sys.exit(1)
        click.echo(f"Content: {node.content[:100]}...")
        if not click.confirm("Delete this memory?"):
            return

    success = ctx.delete_memory(memory_id, agent_id=agent)
    if not success:
        click.echo(f"Memory not found: {memory_id}", err=True)
        sys.exit(1)
    click.echo(f"Deleted: {memory_id}")


@cli.command("update")
@click.argument("memory_id")
@click.option("--content", help="New content (triggers re-embedding)")
@click.option("--tags", multiple=True, help="New tags (replaces existing, repeatable)")
@click.option("--salience", type=float, help="New salience 0-1")
@click.option("--visibility", type=click.Choice(["shared", "private", "thread"]))
@click.option("--agent", help="Agent ID for access check")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def update_memory(memory_id, content, tags, salience, visibility, agent, as_json):
    """Update a memory's content or metadata."""
    ctx = get_cortex()

    vis = Visibility(visibility) if visibility else None
    tag_list = list(tags) if tags else None

    updated = ctx.update_memory(
        memory_id=memory_id,
        content=content,
        tags=tag_list,
        salience=salience,
        visibility=vis,
        agent_id=agent,
    )

    if updated is None:
        click.echo(f"Memory not found: {memory_id}", err=True)
        sys.exit(1)

    if as_json:
        _json_out({
            "id": updated.id,
            "content": updated.content,
            "type": updated.metadata.memory_type.value,
            "salience": round(updated.metadata.salience, 3),
            "tags": updated.metadata.tags,
        })
    else:
        click.echo(f"Updated: {updated.id}")
        click.echo(f"  Salience: {updated.metadata.salience:.2f}")
        click.echo(f"  Tags:     {', '.join(updated.metadata.tags) if updated.metadata.tags else 'none'}")


# =============================================================================
# Share (change visibility)
# =============================================================================

@cli.command("share")
@click.argument("memory_id")
@click.argument("visibility", type=click.Choice(["shared", "private", "thread"]))
@click.option("--agent", help="Agent ID (must be owner)")
def share_memory(memory_id, visibility, agent):
    """Change a memory's visibility (only owner can change)."""
    ctx = get_cortex()
    vis = Visibility(visibility)
    updated = ctx.share_memory(memory_id, new_visibility=vis, agent_id=agent)
    if updated is None:
        click.echo(f"Not found or not authorized: {memory_id}", err=True)
        sys.exit(1)
    click.echo(f"Visibility changed: {memory_id} -> {visibility}")


# =============================================================================
# Associate
# =============================================================================

@cli.command()
@click.argument("source_id")
@click.argument("target_id")
@click.argument("link_type", type=click.Choice(
    ["temporal", "causal", "semantic", "affective", "contextual",
     "contradicts", "supports", "derived_from", "part_of"],
))
@click.option("--weight", type=float, default=0.5, help="Link weight 0-1")
@click.option("--evidence", help="Why this link exists")
def associate(source_id, target_id, link_type, weight, evidence):
    """Create a link between two memories."""
    ctx = get_cortex()

    link_id = ctx.associate(
        source_id=source_id,
        target_id=target_id,
        link_type=LinkType(link_type),
        weight=weight,
        evidence=evidence,
    )

    if link_id is None:
        click.echo("Error: one or both memory IDs not found.", err=True)
        sys.exit(1)

    click.echo(f"Link created: {link_id}")
    click.echo(f"  {source_id} --[{link_type} w={weight}]--> {target_id}")


# =============================================================================
# Episodes
# =============================================================================

@cli.group()
def episode():
    """Episode management commands."""
    pass


@episode.command("start")
@click.option("--title", help="Episode title")
@click.option("--session", help="Session ID")
@click.option("--agent", default="CLAUDE", help="Agent ID")
def episode_start(title, session, agent):
    """Start a new episode."""
    ctx = get_cortex()
    ep = ctx.episode_start(title=title, session_id=session, agent_id=agent)
    click.echo(f"Episode started: {ep.id}")
    if ep.title:
        click.echo(f"  Title: {ep.title}")


@episode.command("step")
@click.argument("episode_id")
@click.argument("memory_id")
@click.option("--role", default="event", type=click.Choice(["event", "context", "outcome", "reflection"]))
def episode_step(episode_id, memory_id, role):
    """Add a step to an episode."""
    ctx = get_cortex()
    step = ctx.episodes.add_step(episode_id=episode_id, memory_id=memory_id, role=role)
    if step is None:
        click.echo("Error: episode not found.", err=True)
        sys.exit(1)
    click.echo(f"Step added at position {step.position} (role: {step.role})")


@episode.command("end")
@click.argument("episode_id")
@click.option("--summary", help="Episode summary")
@click.option("--valence", default="neutral", type=click.Choice(["positive", "negative", "neutral", "mixed"]))
def episode_end(episode_id, summary, valence):
    """End an episode."""
    ctx = get_cortex()
    ep = ctx.episode_end(
        episode_id=episode_id,
        summary=summary,
        valence=EmotionalValence(valence),
    )
    if ep is None:
        click.echo("Error: episode not found.", err=True)
        sys.exit(1)
    click.echo(f"Episode ended: {ep.id} ({len(ep.steps)} steps)")

@episode.command("list")
@click.option("--limit", default=10, help="Max episodes to show")
@click.option("--agent", help="Filter by agent ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def episode_list(limit, agent, as_json):
    """List recent episodes."""
    ctx = get_cortex()
    episodes = ctx.list_episodes(limit=limit, agent_id=agent)

    if as_json:
        _json_out({
            "count": len(episodes),
            "episodes": [
                {
                    "id": ep.id,
                    "title": ep.title,
                    "steps": len(ep.steps),
                    "valence": ep.overall_valence.value,
                    "started_at": ep.started_at.isoformat() if ep.started_at else None,
                }
                for ep in episodes
            ],
        })
        return

    if not episodes:
        click.echo("No episodes found.")
        return

    click.echo(f"Found {len(episodes)} episodes:\n")
    for i, ep in enumerate(episodes, 1):
        title = ep.title or "(untitled)"
        started = ep.started_at.strftime("%Y-%m-%d %H:%M") if ep.started_at else "n/a"
        click.echo(f"  {i}. {title}")
        click.echo(f"     id: {ep.id}  steps: {len(ep.steps)}  valence: {ep.overall_valence.value}  started: {started}")
        click.echo()


@episode.command("get")
@click.argument("episode_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def episode_get(episode_id, as_json):
    """Get an episode by ID with its steps."""
    ctx = get_cortex()
    ep = ctx.get_episode(episode_id)
    if ep is None:
        click.echo(f"Episode not found: {episode_id}", err=True)
        sys.exit(1)

    memories = ctx.get_episode_memories(episode_id)
    mem_map = {m.id: m for m in memories}

    if as_json:
        _json_out({
            "id": ep.id,
            "title": ep.title,
            "agent_id": ep.agent_id,
            "session_id": ep.session_id,
            "valence": ep.overall_valence.value,
            "peak_arousal": ep.peak_arousal,
            "consolidated": ep.consolidated,
            "started_at": ep.started_at.isoformat() if ep.started_at else None,
            "ended_at": ep.ended_at.isoformat() if ep.ended_at else None,
            "steps": [
                {
                    "position": step.position,
                    "memory_id": step.memory_id,
                    "role": step.role,
                    "content": mem_map[step.memory_id].content if step.memory_id in mem_map else None,
                }
                for step in ep.steps
            ],
        })
        return

    title = ep.title or "(untitled)"
    click.echo(f"Episode: {title}")
    click.echo(f"  ID:          {ep.id}")
    click.echo(f"  Agent:       {ep.agent_id}")
    click.echo(f"  Valence:     {ep.overall_valence.value}")
    click.echo(f"  Arousal:     {ep.peak_arousal:.2f}")
    click.echo(f"  Consolidated:{ep.consolidated}")
    if ep.started_at:
        click.echo(f"  Started:     {ep.started_at.strftime('%Y-%m-%d %H:%M')}")
    if ep.ended_at:
        click.echo(f"  Ended:       {ep.ended_at.strftime('%Y-%m-%d %H:%M')}")

    click.echo(f"\nSteps ({len(ep.steps)}):")
    for step in ep.steps:
        mem = mem_map.get(step.memory_id)
        if mem:
            preview = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content
        else:
            preview = "(memory not found)"
        click.echo(f"  [{step.position}] ({step.role}) {preview}")
        click.echo(f"      memory_id: {step.memory_id}")


# =============================================================================
# Intentions (prospective memory)
# =============================================================================

@cli.group()
def intention():
    """Intention (prospective memory) management commands."""
    pass


@intention.command("add")
@click.argument("content")
@click.option("--tags", multiple=True, help="Tags (repeatable)")
@click.option("--salience", type=float, default=0.7, help="Importance 0-1")
@click.option("--agent", default="CLAUDE", help="Agent ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def intention_add(content, tags, salience, agent, as_json):
    """Store a new intention (future TODO / prospective memory)."""
    ctx = get_cortex()
    node = ctx.store_intention(
        content=content,
        tags=list(tags) if tags else None,
        agent_id=agent,
        salience=salience,
    )

    if as_json:
        _json_out({
            "id": node.id,
            "content": node.content,
            "salience": round(node.metadata.salience, 3),
            "tags": node.metadata.tags,
            "agent_id": node.metadata.agent_id,
        })
        return

    preview = content[:80] + "..." if len(content) > 80 else content
    click.echo(f"Intention stored: {node.id}")
    click.echo(f"  {preview}")


@intention.command("list")
@click.option("--agent", help="Filter by agent ID")
@click.option("--min-salience", type=float, default=0.3, help="Minimum salience")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def intention_list(agent, min_salience, as_json):
    """List pending intentions."""
    ctx = get_cortex()
    intentions = ctx.list_intentions(agent_id=agent, min_salience=min_salience)

    if as_json:
        _json_out({
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
        })
        return

    if not intentions:
        click.echo("No pending intentions.")
        return

    click.echo(f"Pending intentions ({len(intentions)}):\n")
    for i, node in enumerate(intentions, 1):
        preview = node.content[:100] + "..." if len(node.content) > 100 else node.content
        click.echo(f"  {i}. [sal={node.metadata.salience:.2f}] {preview}")
        click.echo(f"     id: {node.id}")
        click.echo()


@intention.command("resolve")
@click.argument("memory_id")
def intention_resolve(memory_id):
    """Mark an intention as resolved."""
    ctx = get_cortex()
    success = ctx.resolve_intention(memory_id)
    if not success:
        click.echo(f"Intention not found: {memory_id}", err=True)
        sys.exit(1)
    click.echo(f"Resolved: {memory_id}")




# =============================================================================
# Sessions
# =============================================================================

@cli.group()
def session():
    """Session management commands."""
    pass


@session.command("save")
@click.argument("summary")
@click.option("--discovery", multiple=True, help="Key discovery (repeatable)")
@click.option("--todo", multiple=True, help="Unfinished business (repeatable)")
@click.option("--priority", type=click.Choice(["HIGH", "MEDIUM", "LOW"]), default="MEDIUM")
@click.option("--type", "session_type", type=click.Choice(["orientation", "technical", "emotional", "task"]), default="technical")
def session_save(summary, discovery, todo, priority, session_type):
    """Save a session note."""
    ctx = get_cortex()

    parts = [f"SESSION SUMMARY: {summary}"]
    if discovery:
        parts.append("\nKEY DISCOVERIES:")
        for d in discovery:
            parts.append(f"  - {d}")
    if todo:
        parts.append("\nUNFINISHED BUSINESS:")
        for t in todo:
            parts.append(f"  - {t}")

    content = "\n".join(parts)

    node = ctx.remember(
        content=content,
        memory_type=MemoryType.EPISODIC,
        tags=["session_note", f"priority:{priority}", f"session_type:{session_type}"],
        salience={"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.4}.get(priority, 0.7),
    )

    if node is None:
        click.echo("Session note gated out (duplicate?).", err=True)
        sys.exit(1)

    click.echo(f"Session saved: {node.id} (priority: {priority})")


@session.command("recall")
@click.option("--hours", default=168, help="Lookback hours (default: 168 = 1 week)")
@click.option("--priority", type=click.Choice(["HIGH", "MEDIUM", "LOW"]))
@click.option("--limit", default=10, help="Max sessions to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def session_recall(hours, priority, limit, as_json):
    """Recall previous session notes."""
    ctx = get_cortex()
    since = datetime.now() - timedelta(hours=hours)
    nodes = ctx.graph.get_nodes_since(since)

    sessions = []
    for node in nodes:
        if "session_note" not in node.metadata.tags:
            continue
        if priority and f"priority:{priority}" not in node.metadata.tags:
            continue
        sessions.append(node)

    sessions.sort(key=lambda n: n.created_at, reverse=True)
    sessions = sessions[:limit]

    if as_json:
        _json_out({
            "count": len(sessions),
            "sessions": [
                {
                    "id": s.id,
                    "content": s.content,
                    "priority": next((t.split(":")[1] for t in s.metadata.tags if t.startswith("priority:")), "MEDIUM"),
                    "created_at": s.created_at.isoformat(),
                }
                for s in sessions
            ],
        })
        return

    if not sessions:
        click.echo("No session notes found.")
        return

    click.echo(f"Found {len(sessions)} session notes:\n")
    for s in sessions:
        pri = next((t.split(":")[1] for t in s.metadata.tags if t.startswith("priority:")), "MEDIUM")
        click.echo(f"--- [{pri}] {s.created_at.strftime('%Y-%m-%d %H:%M')} ---")
        click.echo(s.content)
        click.echo(f"  id: {s.id}\n")


# =============================================================================
# Agents
# =============================================================================

@cli.group()
def agents():
    """Agent management commands."""
    pass


@agents.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def agents_list(as_json):
    """List registered agents."""
    ctx = get_cortex()
    agent_list = ctx.graph.list_agents()

    if as_json:
        _json_out({
            "count": len(agent_list),
            "agents": [
                {
                    "id": a.id,
                    "display_name": a.display_name,
                    "generation": a.generation,
                    "lineage": a.lineage,
                    "specialization": a.specialization,
                    "symbol": a.symbol,
                }
                for a in agent_list
            ],
        })
        return

    if not agent_list:
        click.echo("No agents registered.")
        return

    click.echo(f"{len(agent_list)} Registered Agents:\n")
    for a in agent_list:
        click.echo(f"  {a.symbol} {a.id} ({a.display_name})")
        click.echo(f"    Gen {a.generation} | {a.lineage} | {a.specialization}")


@agents.command("register")
@click.argument("agent_id")
@click.argument("display_name")
@click.option("--generation", type=int, default=0)
@click.option("--lineage", default="")
@click.option("--specialization", default="")
@click.option("--symbol", default="A")
@click.option("--color", default="#888888")
def agents_register(agent_id, display_name, generation, lineage, specialization, symbol, color):
    """Register a new agent."""
    ctx = get_cortex()
    profile = AgentProfile(
        id=agent_id,
        display_name=display_name,
        generation=generation,
        lineage=lineage,
        specialization=specialization,
        color=color,
        symbol=symbol,
    )
    ctx.graph.register_agent(profile)
    click.echo(f"Agent registered: {symbol} {agent_id} ({display_name})")


# =============================================================================
# Health
# =============================================================================

@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def health(as_json):
    """Memory system health report."""
    ctx = get_cortex()
    s = ctx.stats()

    promotions = ctx.executive.run_promotion_sweep()
    intentions = ctx.executive.get_pending_intentions()

    if as_json:
        _json_out({
            "memories": s["nodes"],
            "links": s["links"],
            "episodes": s["episodes"],
            "schemas": s["schemas"],
            "by_type": s.get("memory_types", {}),
            "by_layer": s.get("layers", {}),
            "promotions": promotions,
            "pending_intentions": len(intentions),
        })
        return

    click.echo("CerebroCortex Health Report")
    click.echo("=" * 40)
    click.echo(f"  Memories:    {s['nodes']}")
    click.echo(f"  Links:       {s['links']}")
    click.echo(f"  Episodes:    {s['episodes']}")
    click.echo(f"  Schemas:     {s['schemas']}")

    click.echo("\nBy Type:")
    for t, c in s.get("memory_types", {}).items():
        if c > 0:
            click.echo(f"  {t:15s} {c}")

    click.echo("\nBy Layer:")
    for l, c in s.get("layers", {}).items():
        if c > 0:
            click.echo(f"  {l:15s} {c}")

    if promotions:
        click.echo("\nPromotions:")
        for layer, count in promotions.items():
            click.echo(f"  -> {layer}: {count}")

    click.echo(f"\nPending intentions: {len(intentions)}")


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def backfill(as_json):
    """Backfill ChromaDB vector store from GraphStore.

    Finds memories that exist in SQLite/igraph but are missing from
    ChromaDB and inserts them. Safe to run multiple times.
    """
    ctx = get_cortex()
    if not as_json:
        click.echo("Scanning for memories missing from vector store...")
    result = ctx.backfill_vector_store()
    if as_json:
        _json_out(result)
    else:
        total = result.get("total", 0)
        errors = result.get("errors", 0)
        if total == 0:
            click.echo("All memories already in vector store — nothing to do.")
        else:
            click.echo(f"Backfilled {total} memories into ChromaDB:")
            for k, v in result.items():
                if k not in ("total", "errors"):
                    click.echo(f"  {k}: {v}")
            if errors:
                click.echo(f"  errors: {errors}")


# =============================================================================
# Dream Engine
# =============================================================================

@cli.group()
def dream():
    """Dream Engine consolidation commands."""
    pass


@dream.command("run")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def dream_run(as_json):
    """Run a dream consolidation cycle (all 6 phases)."""
    ctx = get_cortex()

    from cerebro.engines.dream import DreamEngine
    try:
        from cerebro.utils.llm import LLMClient
        llm = LLMClient()
    except Exception:
        llm = None
        click.echo("Warning: No LLM available. LLM-assisted phases will be skipped.", err=True)

    engine = DreamEngine(ctx, llm_client=llm)
    click.echo("Starting dream cycle...")

    report = engine.run_cycle()

    if as_json:
        _json_out(report.to_dict())
        return

    click.echo(f"\nDream Cycle Complete")
    click.echo("=" * 40)
    click.echo(f"  Duration:     {report.total_duration_seconds:.1f}s")
    click.echo(f"  Episodes:     {report.episodes_consolidated} consolidated")
    click.echo(f"  LLM calls:    {report.total_llm_calls}")
    click.echo(f"  Success:      {report.success}")
    click.echo(f"\nPhases:")
    for p in report.phases:
        status = "OK" if p.success else "FAIL"
        click.echo(f"  {p.phase.value:30s} [{status}] {p.duration_seconds:.1f}s - {p.notes}")


@dream.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def dream_status(as_json):
    """Show last dream cycle report."""
    ctx = get_cortex()

    from cerebro.engines.dream import DreamEngine
    engine = DreamEngine(ctx)

    if engine.last_report is None:
        if as_json:
            _json_out({"status": "idle", "last_report": None})
        else:
            click.echo("No dream cycles have run yet.")
        return

    report = engine.last_report
    if as_json:
        _json_out(report.to_dict())
    else:
        click.echo(f"Last dream cycle: {report.started_at.strftime('%Y-%m-%d %H:%M')}")
        click.echo(f"  Duration: {report.total_duration_seconds:.1f}s")
        click.echo(f"  Success:  {report.success}")



# =============================================================================
# Graph Exploration (Phase D)
# =============================================================================

@cli.group()
def graph():
    """Graph exploration commands."""
    pass


@graph.command("path")
@click.argument("source_id")
@click.argument("target_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def graph_path(source_id, target_id, as_json):
    """Find path between two memories."""
    ctx = get_cortex()
    try:
        result = ctx.find_path(source_id, target_id)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        _json_out(result)
        return

    if not result or not result.get("path"):
        click.echo(f"No path found between {source_id} and {target_id}.")
        return

    path = result["path"]
    click.echo(f"Path ({len(path)} hops):\n")
    for i, node_id in enumerate(path):
        prefix = "  -> " if i > 0 else "     "
        click.echo(f"{prefix}{node_id}")


@graph.command("common")
@click.argument("id_a")
@click.argument("id_b")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def graph_common(id_a, id_b, as_json):
    """Find common neighbors of two memories."""
    ctx = get_cortex()
    try:
        result = ctx.get_common_neighbors(id_a, id_b)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        _json_out(result)
        return

    neighbors = result if isinstance(result, list) else result.get("neighbors", [])
    if not neighbors:
        click.echo(f"No common neighbors between {id_a} and {id_b}.")
        return

    click.echo(f"Common neighbors ({len(neighbors)}):\n")
    for n in neighbors:
        if isinstance(n, dict):
            click.echo(f"  {n.get('id', n)}")
        else:
            click.echo(f"  {n}")


# =============================================================================
# Schema (Phase D)
# =============================================================================

@cli.group()
def schema():
    """Schema management commands."""
    pass


@schema.command("create")
@click.argument("content")
@click.option("--source", "source_ids", multiple=True, required=True, help="Source memory ID (repeatable)")
@click.option("--tags", multiple=True, help="Tags (repeatable)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def schema_create(content, source_ids, tags, as_json):
    """Create a schema from source memories."""
    ctx = get_cortex()
    try:
        node = ctx.create_schema(
            content=content,
            source_ids=list(source_ids),
            tags=list(tags) if tags else None,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        _json_out({
            "id": node.id,
            "content": node.content,
            "type": node.metadata.memory_type.value,
            "salience": round(node.metadata.salience, 3),
            "tags": node.metadata.tags,
            "source_ids": list(source_ids),
        })
        return

    click.echo(f"Schema created: {node.id}")
    click.echo(f"  Content:  {content[:100]}{'...' if len(content) > 100 else ''}")
    click.echo(f"  Sources:  {len(source_ids)}")
    click.echo(f"  Salience: {node.metadata.salience:.2f}")
    if node.metadata.tags:
        click.echo(f"  Tags:     {', '.join(node.metadata.tags)}")


@schema.command("list")
@click.option("--agent", help="Filter by agent ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def schema_list(agent, as_json):
    """List schemas."""
    ctx = get_cortex()
    schemas = ctx.list_schemas(agent_id=agent)

    if as_json:
        _json_out({
            "count": len(schemas),
            "schemas": [
                {
                    "id": node.id,
                    "content": node.content,
                    "salience": round(node.metadata.salience, 3),
                    "tags": node.metadata.tags,
                    "agent_id": node.metadata.agent_id,
                    "created_at": node.created_at.isoformat(),
                }
                for node in schemas
            ],
        })
        return

    if not schemas:
        click.echo("No schemas found.")
        return

    click.echo(f"Schemas ({len(schemas)}):\n")
    for i, node in enumerate(schemas, 1):
        preview = node.content[:100] + "..." if len(node.content) > 100 else node.content
        click.echo(f"  {i}. [sal={node.metadata.salience:.2f}] {preview}")
        if node.metadata.tags:
            click.echo(f"     tags: {', '.join(node.metadata.tags)}")
        click.echo(f"     id: {node.id}")
        click.echo()


# =============================================================================
# Procedure (Phase D)
# =============================================================================

@cli.group()
def procedure():
    """Procedural memory management commands."""
    pass


@procedure.command("add")
@click.argument("content")
@click.option("--tags", multiple=True, help="Tags (repeatable)")
@click.option("--derived-from", "derived_from", multiple=True, help="Source memory ID (repeatable)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def procedure_add(content, tags, derived_from, as_json):
    """Store a procedural memory."""
    ctx = get_cortex()
    try:
        node = ctx.store_procedure(
            content=content,
            tags=list(tags) if tags else None,
            derived_from=list(derived_from) if derived_from else None,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        _json_out({
            "id": node.id,
            "content": node.content,
            "type": node.metadata.memory_type.value,
            "salience": round(node.metadata.salience, 3),
            "tags": node.metadata.tags,
            "derived_from": list(derived_from) if derived_from else [],
        })
        return

    preview = content[:80] + "..." if len(content) > 80 else content
    click.echo(f"Procedure stored: {node.id}")
    click.echo(f"  {preview}")
    if node.metadata.tags:
        click.echo(f"  Tags: {', '.join(node.metadata.tags)}")


@procedure.command("list")
@click.option("--agent", help="Filter by agent ID")
@click.option("--min-salience", type=float, default=0.0, help="Minimum salience")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def procedure_list(agent, min_salience, as_json):
    """List procedural memories."""
    ctx = get_cortex()
    procedures = ctx.list_procedures(agent_id=agent, min_salience=min_salience)

    if as_json:
        _json_out({
            "count": len(procedures),
            "procedures": [
                {
                    "id": node.id,
                    "content": node.content,
                    "salience": round(node.metadata.salience, 3),
                    "tags": node.metadata.tags,
                    "agent_id": node.metadata.agent_id,
                    "created_at": node.created_at.isoformat(),
                }
                for node in procedures
            ],
        })
        return

    if not procedures:
        click.echo("No procedures found.")
        return

    click.echo(f"Procedures ({len(procedures)}):\n")
    for i, node in enumerate(procedures, 1):
        preview = node.content[:100] + "..." if len(node.content) > 100 else node.content
        click.echo(f"  {i}. [sal={node.metadata.salience:.2f}] {preview}")
        if node.metadata.tags:
            click.echo(f"     tags: {', '.join(node.metadata.tags)}")
        click.echo(f"     id: {node.id}")
        click.echo()


@procedure.command("outcome")
@click.argument("procedure_id")
@click.option("--success", "outcome", flag_value="success", help="Record successful outcome")
@click.option("--failure", "outcome", flag_value="failure", help="Record failed outcome")
def procedure_outcome(procedure_id, outcome):
    """Record outcome for a procedural memory."""
    if outcome is None:
        click.echo("Error: must specify --success or --failure.", err=True)
        sys.exit(1)

    ctx = get_cortex()
    success = outcome == "success"
    try:
        result = ctx.record_procedure_outcome(procedure_id, success=success)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Outcome recorded for {procedure_id}: {'success' if success else 'failure'}")


# =============================================================================
# Emotions (Phase D)
# =============================================================================

@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def emotions(as_json):
    """Show emotional valence breakdown."""
    ctx = get_cortex()
    try:
        summary = ctx.get_emotional_summary()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        _json_out(summary)
        return

    click.echo("Emotional Valence Breakdown")
    click.echo("=" * 40)
    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, dict):
                click.echo(f"\n  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        click.echo(f"    {k:15s} {v:.3f}")
                    else:
                        click.echo(f"    {k:15s} {v}")
            elif isinstance(value, float):
                click.echo(f"  {key:15s} {value:.3f}")
            else:
                click.echo(f"  {key:15s} {value}")
    else:
        click.echo(f"  {summary}")

# =============================================================================
# Import
# =============================================================================

@cli.group("import")
def import_cmd():
    """Import memories from external sources."""
    pass


@import_cmd.command("neocortex")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def import_neocortex(file, as_json):
    """Import from a Neo-Cortex JSON export file."""
    from cerebro.migration.neo_cortex_import import NeoCortexImporter

    ctx = get_cortex()
    importer = NeoCortexImporter(ctx)

    click.echo(f"Importing from {file}...")
    report = importer.import_file(file)

    if as_json:
        _json_out(report.to_dict())
        return

    click.echo(f"\nImport Complete ({report.duration_seconds:.1f}s)")
    click.echo("=" * 40)
    click.echo(f"  Imported:  {report.memories_imported}")
    click.echo(f"  Skipped:   {report.memories_skipped}")
    click.echo(f"  Links:     {report.links_created}")
    click.echo(f"  Agents:    {report.agents_registered}")
    if report.errors:
        click.echo(f"  Errors:    {len(report.errors)}")
        for err in report.errors[:5]:
            click.echo(f"    - {err}")


@import_cmd.command("json")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def import_json(file, as_json):
    """Import from a generic JSON file."""
    from cerebro.migration.json_import import JSONImporter

    ctx = get_cortex()
    importer = JSONImporter(ctx)

    click.echo(f"Importing from {file}...")
    report = importer.import_file(file)

    if as_json:
        _json_out(report.to_dict())
        return

    click.echo(f"\nImport Complete ({report.duration_seconds:.1f}s)")
    click.echo("=" * 40)
    click.echo(f"  Imported:  {report.memories_imported}")
    click.echo(f"  Skipped:   {report.memories_skipped}")
    if report.errors:
        click.echo(f"  Errors:    {len(report.errors)}")
        for err in report.errors[:5]:
            click.echo(f"    - {err}")


@import_cmd.command("markdown")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def import_markdown(file, as_json):
    """Import from a Markdown file."""
    from cerebro.migration.markdown_import import MarkdownImporter

    ctx = get_cortex()
    importer = MarkdownImporter(ctx)

    click.echo(f"Importing from {file}...")
    report = importer.import_file(file)

    if as_json:
        _json_out(report.to_dict())
        return

    click.echo(f"\nImport Complete ({report.duration_seconds:.1f}s)")
    click.echo("=" * 40)
    click.echo(f"  Imported:  {report.memories_imported}")
    click.echo(f"  Skipped:   {report.memories_skipped}")
    if report.errors:
        click.echo(f"  Errors:    {len(report.errors)}")
        for err in report.errors[:5]:
            click.echo(f"    - {err}")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    cli()


if __name__ == "__main__":
    main()
