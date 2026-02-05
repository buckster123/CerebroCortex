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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def recall(query, results, memory_type, agent, min_salience, as_json):
    """Search and retrieve memories."""
    ctx = get_cortex()

    memory_types = [MemoryType(memory_type)] if memory_type else None

    found = ctx.recall(
        query=query,
        top_k=results,
        memory_types=memory_types,
        agent_id=agent,
        min_salience=min_salience,
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
