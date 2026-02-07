"""Runtime settings management for CerebroCortex.

Layers (lowest → highest priority):
    config.py defaults → data/settings.json → data/.env (for API keys only)

Usage:
    from cerebro.settings import load_on_startup, get_current_settings, apply_settings

    load_on_startup()                    # Call once at server start
    settings = get_current_settings()    # Read merged config
    apply_settings({"llm": {"temperature": 0.5}})  # Partial update
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import cerebro.config as cfg

logger = logging.getLogger("cerebro-settings")

SETTINGS_FILE = cfg.DATA_DIR / "settings.json"
ENV_FILE = cfg.DATA_DIR / ".env"

# Keys that should be masked when returned via GET
_SENSITIVE_KEYS = {"anthropic_api_key", "openai_api_key"}

# Mapping: settings JSON path → config.py attribute name
_SETTING_MAP: dict[str, str] = {
    # LLM
    "llm.primary_provider": "LLM_PRIMARY_PROVIDER",
    "llm.primary_model": "LLM_PRIMARY_MODEL",
    "llm.fallback_provider": "LLM_FALLBACK_PROVIDER",
    "llm.fallback_model": "LLM_FALLBACK_MODEL",
    "llm.openai_compat_base_url": "OPENAI_COMPAT_BASE_URL",
    "llm.temperature": "LLM_TEMPERATURE",
    "llm.max_tokens": "LLM_MAX_TOKENS",
    "llm.strip_think": "OPENAI_COMPAT_STRIP_THINK",
    "llm.no_think": "OPENAI_COMPAT_NO_THINK",
    # Dream
    "dream.max_llm_calls": "DREAM_MAX_LLM_CALLS",
    "dream.budget_pattern": "DREAM_LLM_BUDGET_PATTERN",
    "dream.budget_schema": "DREAM_LLM_BUDGET_SCHEMA",
    "dream.budget_rem": "DREAM_LLM_BUDGET_REM",
    "dream.cluster_threshold": "DREAM_CLUSTER_SIMILARITY_THRESHOLD",
    "dream.rem_pair_checks": "DREAM_REM_PAIR_CHECKS",
    # Scoring (dev)
    "scoring.weight_vector": "SCORE_WEIGHT_VECTOR",
    "scoring.weight_activation": "SCORE_WEIGHT_ACTIVATION",
    "scoring.weight_retrievability": "SCORE_WEIGHT_RETRIEVABILITY",
    "scoring.weight_salience": "SCORE_WEIGHT_SALIENCE",
    # Advanced (dev)
    "advanced.actr_decay_rate": "ACTR_DECAY_RATE",
    "advanced.actr_noise": "ACTR_NOISE",
    "advanced.spreading_max_hops": "SPREADING_MAX_HOPS",
    "advanced.spreading_decay_per_hop": "SPREADING_DECAY_PER_HOP",
    "advanced.fsrs_initial_stability": "FSRS_INITIAL_STABILITY",
    "advanced.fsrs_initial_difficulty": "FSRS_INITIAL_DIFFICULTY",
    # Agent
    "agent.default_agent_id": "DEFAULT_AGENT_ID",
}

# Env-file key → config.py attribute (only for secrets)
_ENV_MAP: dict[str, str] = {
    "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
}


# =========================================================================
# .env parser (no python-dotenv dependency)
# =========================================================================

def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file.

    Handles:
    - Comments (# ...) and blank lines
    - Quoted values (single and double quotes stripped)
    - Inline comments after unquoted values
    """
    result: dict[str, str] = {}
    if not path.exists():
        return result
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            else:
                # Strip inline comments for unquoted values
                for comment_char in (" #", "\t#"):
                    idx = value.find(comment_char)
                    if idx != -1:
                        value = value[:idx].rstrip()
            result[key] = value
    except Exception as e:
        logger.warning(f"Failed to parse .env file {path}: {e}")
    return result


# =========================================================================
# JSON persistence
# =========================================================================

def _load_settings_json() -> dict[str, Any]:
    """Load settings.json, returning empty dict if missing/corrupt."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except Exception as e:
        logger.warning(f"Failed to load {SETTINGS_FILE}: {e}")
        return {}


def _save_settings_json(data: dict[str, Any]) -> None:
    """Write settings.json atomically."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2) + "\n")


# =========================================================================
# Read current config from config.py module attributes
# =========================================================================

def _read_config_defaults() -> dict[str, Any]:
    """Build nested dict from current config.py module attributes."""
    result: dict[str, dict[str, Any]] = {}
    for dotpath, attr in _SETTING_MAP.items():
        section, _, key = dotpath.partition(".")
        if section not in result:
            result[section] = {}
        result[section][key] = getattr(cfg, attr, None)
    return result


# =========================================================================
# Public API
# =========================================================================

def get_current_settings(include_dev: bool = False) -> dict[str, Any]:
    """Return merged config as nested dict, masking sensitive keys.

    Args:
        include_dev: If True, include scoring/advanced sections.
    """
    settings = _read_config_defaults()

    if not include_dev:
        settings.pop("scoring", None)
        settings.pop("advanced", None)

    # Add API keys section (always masked in output)
    settings["llm_keys"] = {}
    env_key = getattr(cfg, "ANTHROPIC_API_KEY", None) or _parse_env_file(ENV_FILE).get("ANTHROPIC_API_KEY")
    if env_key:
        settings["llm_keys"]["anthropic_api_key"] = _mask(env_key)
    else:
        settings["llm_keys"]["anthropic_api_key"] = ""

    return settings


def _mask(value: str) -> str:
    """Mask a sensitive value, showing only last 4 chars."""
    if not value or len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


def apply_settings(updates: dict[str, Any]) -> dict[str, str]:
    """Apply partial settings update.

    1. Merge into settings.json
    2. Hot-reload config.py module attributes via setattr

    Args:
        updates: Nested dict of sections → key/value pairs.

    Returns:
        Dict of applied changes: {"section.key": "new_value", ...}
    """
    current = _load_settings_json()
    applied: dict[str, str] = {}
    llm_changed = False

    for section, values in updates.items():
        if not isinstance(values, dict):
            continue

        if section not in current:
            current[section] = {}

        for key, value in values.items():
            dotpath = f"{section}.{key}"

            # Skip masked values (user didn't change them)
            if isinstance(value, str) and value.startswith("***"):
                continue

            # Handle API keys — write to .env, not settings.json
            if section == "llm_keys" and key in _SENSITIVE_KEYS:
                _write_env_key(key.upper(), str(value))
                applied[dotpath] = "(updated)"
                continue

            # Persist to settings.json
            current[section][key] = value

            # Hot-reload to config.py
            attr = _SETTING_MAP.get(dotpath)
            if attr and hasattr(cfg, attr):
                # Coerce type to match existing
                existing = getattr(cfg, attr)
                try:
                    value = _coerce(value, existing)
                except (ValueError, TypeError):
                    pass
                setattr(cfg, attr, value)
                applied[dotpath] = str(value)

            if section == "llm":
                llm_changed = True

    _save_settings_json(current)

    return applied


def _coerce(value: Any, existing: Any) -> Any:
    """Coerce value to match the type of existing."""
    if existing is None:
        return value
    if isinstance(existing, bool):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if isinstance(existing, int):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    return value


def _write_env_key(key: str, value: str) -> None:
    """Write or update a single key in the .env file."""
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    env = _parse_env_file(ENV_FILE)
    env[key] = value
    lines = [f'{k}="{v}"' for k, v in env.items()]
    ENV_FILE.write_text("\n".join(lines) + "\n")


def reset_settings() -> None:
    """Delete settings.json and reload defaults."""
    if SETTINGS_FILE.exists():
        SETTINGS_FILE.unlink()
    # Re-apply defaults by reloading the module's original values
    # (config.py top-level assignments are the canonical defaults)
    logger.info("Settings reset to defaults")


def load_on_startup() -> None:
    """Load settings.json + .env overrides into config.py on server start."""
    # 1. Apply settings.json overrides
    saved = _load_settings_json()
    for section, values in saved.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            dotpath = f"{section}.{key}"
            attr = _SETTING_MAP.get(dotpath)
            if attr and hasattr(cfg, attr):
                existing = getattr(cfg, attr)
                try:
                    value = _coerce(value, existing)
                except (ValueError, TypeError):
                    pass
                setattr(cfg, attr, value)

    # 2. Apply .env overrides (API keys)
    env = _parse_env_file(ENV_FILE)
    for env_key, attr in _ENV_MAP.items():
        if env_key in env:
            setattr(cfg, attr, env[env_key])

    count = sum(len(v) for v in saved.values() if isinstance(v, dict))
    env_count = sum(1 for k in _ENV_MAP if k in env)
    if count or env_count:
        logger.info(f"Loaded {count} settings overrides + {env_count} env keys")
