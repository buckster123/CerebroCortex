"""Tests for the settings manager module."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from cerebro import config as cfg
from cerebro.settings import (
    _coerce,
    _mask,
    _parse_env_file,
    apply_settings,
    get_current_settings,
    load_on_startup,
    reset_settings,
)


# =========================================================================
# .env parser
# =========================================================================

class TestEnvParser:
    def test_basic_key_value(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("FOO=bar\nBAZ=123\n")
        result = _parse_env_file(f)
        assert result == {"FOO": "bar", "BAZ": "123"}

    def test_quoted_values(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text('KEY="hello world"\nSINGLE=\'quoted\'\n')
        result = _parse_env_file(f)
        assert result["KEY"] == "hello world"
        assert result["SINGLE"] == "quoted"

    def test_comments_and_blanks(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("# comment\n\nFOO=bar\n  # another comment\nBAZ=qux\n")
        result = _parse_env_file(f)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_inline_comment(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("KEY=value # inline comment\n")
        result = _parse_env_file(f)
        assert result["KEY"] == "value"

    def test_nonexistent_file(self, tmp_path):
        f = tmp_path / "nope.env"
        result = _parse_env_file(f)
        assert result == {}

    def test_empty_file(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("")
        result = _parse_env_file(f)
        assert result == {}

    def test_no_equals(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("INVALID_LINE\nGOOD=yes\n")
        result = _parse_env_file(f)
        assert result == {"GOOD": "yes"}

    def test_value_with_equals(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("URL=http://host:1234/path?a=b\n")
        result = _parse_env_file(f)
        assert result["URL"] == "http://host:1234/path?a=b"


# =========================================================================
# Settings JSON roundtrip
# =========================================================================

class TestSettingsRoundtrip:
    def test_save_load(self, tmp_path, monkeypatch):
        import cerebro.settings as sm
        monkeypatch.setattr(sm, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(sm, "ENV_FILE", tmp_path / ".env")

        # Apply a setting
        apply_settings({"llm": {"temperature": 0.42}})

        # Verify persisted
        data = json.loads((tmp_path / "settings.json").read_text())
        assert data["llm"]["temperature"] == 0.42

        # Verify hot-reloaded
        assert cfg.LLM_TEMPERATURE == 0.42

    def test_merge_defaults(self, tmp_path, monkeypatch):
        import cerebro.settings as sm
        monkeypatch.setattr(sm, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(sm, "ENV_FILE", tmp_path / ".env")

        settings = get_current_settings()
        assert "llm" in settings
        assert "dream" in settings
        assert "temperature" in settings["llm"]

    def test_dev_sections_hidden_by_default(self, tmp_path, monkeypatch):
        import cerebro.settings as sm
        monkeypatch.setattr(sm, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(sm, "ENV_FILE", tmp_path / ".env")

        settings = get_current_settings(include_dev=False)
        assert "scoring" not in settings
        assert "advanced" not in settings

    def test_dev_sections_included(self, tmp_path, monkeypatch):
        import cerebro.settings as sm
        monkeypatch.setattr(sm, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(sm, "ENV_FILE", tmp_path / ".env")

        settings = get_current_settings(include_dev=True)
        assert "scoring" in settings
        assert "advanced" in settings
        assert "weight_vector" in settings["scoring"]


# =========================================================================
# API key masking
# =========================================================================

class TestMasking:
    def test_mask_long(self):
        val = "sk-ant-api03-abcdef1234"
        masked = _mask(val)
        assert masked.endswith("1234")
        assert masked.startswith("*")
        assert len(masked) == len(val)

    def test_mask_short(self):
        assert _mask("abc") == "****"

    def test_mask_empty(self):
        assert _mask("") == "****"

    def test_api_key_masked_in_output(self, tmp_path, monkeypatch):
        import cerebro.settings as sm
        monkeypatch.setattr(sm, "SETTINGS_FILE", tmp_path / "settings.json")
        env_f = tmp_path / ".env"
        env_f.write_text('ANTHROPIC_API_KEY="sk-ant-test-key-12345678"\n')
        monkeypatch.setattr(sm, "ENV_FILE", env_f)

        settings = get_current_settings()
        key_val = settings["llm_keys"]["anthropic_api_key"]
        assert key_val.endswith("5678")
        assert key_val.startswith("*")
        assert "sk-ant" not in key_val


# =========================================================================
# Apply settings
# =========================================================================

class TestApplySettings:
    def test_setattr_propagation(self, tmp_path, monkeypatch):
        import cerebro.settings as sm
        monkeypatch.setattr(sm, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(sm, "ENV_FILE", tmp_path / ".env")

        original = cfg.DREAM_MAX_LLM_CALLS
        apply_settings({"dream": {"max_llm_calls": 99}})
        assert cfg.DREAM_MAX_LLM_CALLS == 99

        # Restore
        cfg.DREAM_MAX_LLM_CALLS = original

    def test_type_coercion(self):
        assert _coerce("42", 0) == 42
        assert _coerce("0.5", 1.0) == 0.5
        assert _coerce("true", False) is True
        assert _coerce("false", True) is False

    def test_skip_masked_values(self, tmp_path, monkeypatch):
        import cerebro.settings as sm
        monkeypatch.setattr(sm, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(sm, "ENV_FILE", tmp_path / ".env")

        applied = apply_settings({"llm_keys": {"anthropic_api_key": "****abcd"}})
        # Masked value should be skipped
        assert "llm_keys.anthropic_api_key" not in applied


# =========================================================================
# Reset
# =========================================================================

class TestReset:
    def test_reset_deletes_file(self, tmp_path, monkeypatch):
        import cerebro.settings as sm
        sf = tmp_path / "settings.json"
        sf.write_text('{"llm": {"temperature": 0.99}}')
        monkeypatch.setattr(sm, "SETTINGS_FILE", sf)

        reset_settings()
        assert not sf.exists()


# =========================================================================
# Startup loading
# =========================================================================

class TestLoadOnStartup:
    def test_loads_json_and_env(self, tmp_path, monkeypatch):
        import cerebro.settings as sm
        sf = tmp_path / "settings.json"
        sf.write_text('{"llm": {"temperature": 0.33}}')
        monkeypatch.setattr(sm, "SETTINGS_FILE", sf)

        ef = tmp_path / ".env"
        ef.write_text("")
        monkeypatch.setattr(sm, "ENV_FILE", ef)

        original = cfg.LLM_TEMPERATURE
        load_on_startup()
        assert cfg.LLM_TEMPERATURE == 0.33

        # Restore
        cfg.LLM_TEMPERATURE = original

    def test_missing_files_ok(self, tmp_path, monkeypatch):
        import cerebro.settings as sm
        monkeypatch.setattr(sm, "SETTINGS_FILE", tmp_path / "nope.json")
        monkeypatch.setattr(sm, "ENV_FILE", tmp_path / "nope.env")
        # Should not raise
        load_on_startup()
