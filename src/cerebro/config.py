"""Configuration constants for CerebroCortex."""

import os
from pathlib import Path
from cerebro.types import LinkType


# =============================================================================
# Paths
# =============================================================================
def _resolve_data_dir() -> Path:
    """Resolve data directory: CEREBRO_DATA_DIR env var, or ~/.cerebro-cortex/"""
    env_dir = os.environ.get("CEREBRO_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".cerebro-cortex"


def _resolve_web_dir() -> Path:
    """Find web dashboard: package data first, then CWD fallback."""
    pkg_web = Path(__file__).parent / "web"  # src/cerebro/web/ (installed)
    if pkg_web.is_dir():
        return pkg_web
    return Path.cwd() / "web"  # fallback for weird layouts


DATA_DIR = _resolve_data_dir()
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_DB = DATA_DIR / "cerebro.db"
EXPORT_DIR = DATA_DIR / "exports"
WEB_DIR = _resolve_web_dir()

# =============================================================================
# Embedding
# =============================================================================
SBERT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# =============================================================================
# ChromaDB Collections
# =============================================================================
COLLECTION_MEMORIES = "cerebro_memories"
COLLECTION_KNOWLEDGE = "cerebro_knowledge"
COLLECTION_SESSIONS = "cerebro_sessions"
ALL_COLLECTIONS = [COLLECTION_MEMORIES, COLLECTION_KNOWLEDGE, COLLECTION_SESSIONS]

# =============================================================================
# ACT-R Parameters
# =============================================================================
ACTR_DECAY_RATE = 0.5          # d parameter in B(t) = ln(Σ t_k^{-d})
ACTR_B_CONSTANT = 0.0          # additive constant in base-level activation
ACTR_MIN_TIME_SECONDS = 1.0    # floor for time-since-access (avoid division by zero)
ACTR_RETRIEVAL_THRESHOLD = 0.0  # tau - minimum activation for recall
ACTR_NOISE = 0.4               # s - noise parameter in recall probability sigmoid
MAX_STORED_TIMESTAMPS = 50      # keep this many individual timestamps; compress older

# =============================================================================
# FSRS Parameters
# =============================================================================
FSRS_INITIAL_STABILITY = 1.0    # days
FSRS_INITIAL_DIFFICULTY = 5.0   # 1-10 scale
FSRS_MIN_STABILITY = 0.1       # floor
FSRS_MAX_STABILITY = 365.0     # ceiling (1 year)

# =============================================================================
# Combined Recall Scoring Weights
# =============================================================================
SCORE_WEIGHT_VECTOR = 0.35      # semantic similarity from ChromaDB
SCORE_WEIGHT_ACTIVATION = 0.30  # ACT-R activation (base-level + associative)
SCORE_WEIGHT_RETRIEVABILITY = 0.20  # FSRS retrievability (forgetting curve)
SCORE_WEIGHT_SALIENCE = 0.15    # emotional salience metadata

# =============================================================================
# Spreading Activation
# =============================================================================
SPREADING_MAX_HOPS = 2
SPREADING_DECAY_PER_HOP = 0.6
SPREADING_ACTIVATION_THRESHOLD = 0.05
SPREADING_MAX_ACTIVATED = 50

# Link type weights for spreading activation relevance
LINK_TYPE_WEIGHTS: dict[LinkType, float] = {
    LinkType.TEMPORAL: 0.6,
    LinkType.CAUSAL: 0.9,
    LinkType.SEMANTIC: 0.8,
    LinkType.AFFECTIVE: 0.5,
    LinkType.CONTEXTUAL: 0.7,
    LinkType.CONTRADICTS: 0.3,
    LinkType.SUPPORTS: 0.8,
    LinkType.DERIVED_FROM: 0.7,
    LinkType.PART_OF: 0.8,
}

# =============================================================================
# Memory Layers (promotion thresholds)
# =============================================================================
LAYER_CONFIG = {
    "sensory": {
        "decay_half_life_hours": 6,
        "min_salience": 0.1,
        "promotion_access_count": 2,
        "promotion_min_age_hours": None,
    },
    "working": {
        "decay_half_life_hours": 72,
        "min_salience": 0.2,
        "promotion_access_count": 5,
        "promotion_min_age_hours": 24,
    },
    "long_term": {
        "decay_half_life_hours": 720,
        "min_salience": 0.3,
        "promotion_access_count": None,  # promotion via dream engine only
        "promotion_min_age_hours": None,
    },
    "cortex": {
        "decay_half_life_hours": None,  # no decay
        "min_salience": 1.0,
        "promotion_access_count": None,
        "promotion_min_age_hours": None,
    },
}

# =============================================================================
# Dream Engine
# =============================================================================
DREAM_MAX_LLM_CALLS = 20
DREAM_LLM_BUDGET_PATTERN = 12   # Reserve for pattern extraction
DREAM_LLM_BUDGET_SCHEMA = 4    # Reserve for schema formation
DREAM_LLM_BUDGET_REM = 4       # Reserve for REM recombination
DREAM_CLUSTER_SIMILARITY_THRESHOLD = 0.80
DREAM_CLUSTER_MIN_SIZE = 3
DREAM_PRUNING_MIN_AGE_HOURS = 48
DREAM_PRUNING_MAX_SALIENCE = 0.3
DREAM_REM_SAMPLE_SIZE = 20
DREAM_REM_PAIR_CHECKS = 10
DREAM_REM_MIN_CONNECTION_STRENGTH = 0.4

# LLM configuration
LLM_PRIMARY_PROVIDER = "openai_compat"  # "anthropic", "ollama", "openai_compat"
LLM_PRIMARY_MODEL = "qwen/qwen3-4b-2507"
LLM_FALLBACK_PROVIDER = "anthropic"  # Claude API fallback
LLM_FALLBACK_MODEL = "claude-sonnet-4-5-20250929"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024

# OpenAI-compatible API (LM Studio, vLLM, LocalAI, etc.)
OPENAI_COMPAT_BASE_URL = "http://192.168.0.107:1234"
OPENAI_COMPAT_STRIP_THINK = True  # Strip <think>...</think> from responses (Qwen3, etc.)
OPENAI_COMPAT_NO_THINK = True  # Append /no_think to system prompts (disables CoT for Qwen3)

# =============================================================================
# Server
# =============================================================================
MCP_SERVER_NAME = "cerebro-cortex"
try:
    from cerebro import __version__ as MCP_SERVER_VERSION
except ImportError:
    MCP_SERVER_VERSION = "0.1.0"
API_HOST = "0.0.0.0"
API_PORT = 8767

# =============================================================================
# Agent defaults
# =============================================================================
DEFAULT_AGENT_ID = os.environ.get("CEREBRO_AGENT_ID", "CLAUDE")
AGENT_PROFILES = {
    "CLAUDE": {
        "display_name": "Claude",
        "generation": 0,
        "lineage": "Anthropic",
        "specialization": "General assistance",
        "color": "#D4AF37",
        "symbol": "C",
    }
}
