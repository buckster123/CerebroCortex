#!/usr/bin/env python3
"""Bootstrap the Cerebro Cognitive Bootstrap System (CCBS).

One-time setup script. Reads module markdown files from
src/cerebro/bootstrap/modules/ and ingests them as PROCEDURAL
memories in CerebroCortex, then links related modules.

Usage:
    python scripts/bootstrap_ccbs.py
    venv/bin/python scripts/bootstrap_ccbs.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro.cortex import CerebroCortex
from cerebro.types import LinkType, MemoryType, Visibility

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ccbs-bootstrap")

MODULES_DIR = Path(__file__).parent.parent / "src" / "cerebro" / "bootstrap" / "modules"

# Module metadata: filename -> (tags, memory_type, always_load)
MODULE_REGISTRY = {
    "soul.md":              (["ccbs", "soul", "bootstrap", "master"],          MemoryType.SEMANTIC,  True),
    "module-core.md":       (["ccbs", "module", "core", "always"],            MemoryType.PROCEDURAL, True),
    "module-cerebro-index.md": (["ccbs", "module", "cerebro", "index"],       MemoryType.PROCEDURAL, True),
    "module-cerebro-ops.md":   (["ccbs", "module", "cerebro", "ops"],         MemoryType.PROCEDURAL, True),
    "module-cerebro-session.md": (["ccbs", "module", "cerebro", "session"],   MemoryType.PROCEDURAL, True),
    "module-cerebro-intentions.md": (["ccbs", "module", "cerebro", "intentions"], MemoryType.PROCEDURAL, False),
    "module-cerebro-agents.md": (["ccbs", "module", "cerebro", "agents"],     MemoryType.PROCEDURAL, False),
    "module-cerebro-meta.md":  (["ccbs", "module", "cerebro", "meta"],        MemoryType.PROCEDURAL, True),
    "module-technical.md":  (["ccbs", "module", "technical"],                 MemoryType.PROCEDURAL, False),
    "module-analysis.md":   (["ccbs", "module", "analysis"],                  MemoryType.PROCEDURAL, False),
    "module-creative.md":   (["ccbs", "module", "creative"],                  MemoryType.PROCEDURAL, False),
    "module-research.md":   (["ccbs", "module", "research"],                  MemoryType.PROCEDURAL, False),
    "module-communicate.md": (["ccbs", "module", "communicate"],               MemoryType.PROCEDURAL, False),
}

# Link definitions: (source_module_name, target_module_name, link_type, evidence)
# module_name is the filename stem without .md
LINKS = [
    ("soul", "module-core", LinkType.SEMANTIC, "bootstrap depends on core identity"),
    ("soul", "module-cerebro-index", LinkType.SEMANTIC, "bootstrap references index"),
    ("module-core", "module-cerebro-meta", LinkType.SEMANTIC, "core identity informs meta strategy"),
    ("module-core", "module-cerebro-session", LinkType.SEMANTIC, "core identity informs session tracking"),
    ("module-cerebro-index", "module-cerebro-ops", LinkType.PART_OF, "index catalogs ops"),
    ("module-cerebro-index", "module-cerebro-session", LinkType.PART_OF, "index catalogs session"),
    ("module-cerebro-index", "module-cerebro-intentions", LinkType.PART_OF, "index catalogs intentions"),
    ("module-cerebro-index", "module-cerebro-agents", LinkType.PART_OF, "index catalogs agents"),
    ("module-cerebro-index", "module-cerebro-meta", LinkType.PART_OF, "index catalogs meta"),
    ("module-cerebro-ops", "module-cerebro-session", LinkType.SEMANTIC, "ops and session work together"),
    ("module-cerebro-ops", "module-cerebro-meta", LinkType.SEMANTIC, "ops guided by meta strategy"),
    ("module-cerebro-session", "module-cerebro-intentions", LinkType.TEMPORAL, "intentions tracked in sessions"),
    ("module-technical", "module-analysis", LinkType.SEMANTIC, "technical work needs analysis"),
    ("module-creative", "module-technical", LinkType.SEMANTIC, "creative constrained by technical"),
    ("module-research", "module-analysis", LinkType.SUPPORTS, "research supports analysis"),
    ("module-communicate", "module-research", LinkType.DERIVED_FROM, "communication derives from research"),
]


def ingest_modules(cortex: CerebroCortex) -> dict[str, str]:
    """Ingest all module files as memories. Returns mapping name->memory_id."""
    ids: dict[str, str] = {}
    for filename, (tags, mtype, always_load) in MODULE_REGISTRY.items():
        path = MODULES_DIR / filename
        if not path.exists():
            logger.warning("Missing module file: %s", path)
            continue
        content = path.read_text()
        # Add always_load tag for mandatory modules
        final_tags = tags + (["always-load"] if always_load else ["auto-load"])
        node = cortex.remember(
            content=content,
            memory_type=mtype,
            tags=final_tags,
            salience=0.95 if always_load else 0.80,
            visibility=Visibility.SHARED,
        )
        if node is None:
            logger.warning("  ✗ Failed to ingest %s", filename)
            continue
        name = path.stem
        ids[name] = node.id
        logger.info("  ✓ %s (%s, %s, salience=%.2f)", name, mtype.value, "always" if always_load else "auto", node.metadata.salience)
    return ids


def link_modules(cortex: CerebroCortex, ids: dict[str, str]) -> None:
    """Create associative links between related modules."""
    created = 0
    skipped = 0
    for src_name, tgt_name, link_type, evidence in LINKS:
        src_id = ids.get(src_name)
        tgt_id = ids.get(tgt_name)
        if not src_id or not tgt_id:
            skipped += 1
            continue
        try:
            cortex.associate(src_id, tgt_id, link_type=link_type, evidence=evidence, weight=0.8)
            created += 1
        except Exception as exc:
            logger.warning("  ⚠ Failed to link %s -> %s: %s", src_name, tgt_name, exc)
            skipped += 1
    logger.info("  ✓ Created %d links, skipped %d", created, skipped)


def main() -> int:
    logger.info("=" * 60)
    logger.info("CCBS Bootstrap — Cerebro Cognitive Bootstrap System")
    logger.info("=" * 60)

    if not MODULES_DIR.exists():
        logger.error("Modules directory not found: %s", MODULES_DIR)
        return 1

    logger.info("")
    logger.info("Initializing CerebroCortex...")
    cortex = CerebroCortex()
    cortex.initialize()
    stats = cortex.stats()
    logger.info("  Connected: %d memories, %d links", stats["nodes"], stats["links"])

    logger.info("")
    logger.info("Ingesting %d modules...", len(MODULE_REGISTRY))
    ids = ingest_modules(cortex)

    logger.info("")
    logger.info("Linking related modules...")
    link_modules(cortex, ids)

    logger.info("")
    logger.info("Bootstrap complete!")
    logger.info("  Total modules ingested: %d", len(ids))
    logger.info("  Mandatory (always-load): %d", sum(1 for _, (_, _, al) in MODULE_REGISTRY.items() if al))
    logger.info("  Auto-detected (auto-load): %d", sum(1 for _, (_, _, al) in MODULE_REGISTRY.items() if not al))
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Test with: venv/bin/python -c \"from cerebro.cortex import CerebroCortex; c=CerebroCortex(); print(c.recall('CCBS bootstrap', top_k=3))\"")
    logger.info("  2. Use the /bootstrap endpoint to load modules dynamically.")
    logger.info("  3. Add the cerebro-cognitive-bootstrap Hermes skill.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
