"""Tests for the Cerebro Cognitive Bootstrap System (CCBS)."""

import pytest
from cerebro.interfaces.api_server import CognitiveBootstrapAssembler


class TestCognitiveBootstrapAssembler:
    """Unit tests for the query analysis and module assembly logic."""

    @pytest.fixture
    def assembler(self, multi_agent_cortex):
        return CognitiveBootstrapAssembler(multi_agent_cortex)

    def test_detect_triggers_full_load(self, assembler):
        modules, trigger = assembler._detect_triggers("full load, let's go")
        assert trigger == "full load"
        assert modules == []

    def test_detect_triggers_solo(self, assembler):
        modules, trigger = assembler._detect_triggers("solo core quick check")
        assert trigger == "solo"
        assert modules == []

    def test_detect_triggers_debug_mode(self, assembler):
        modules, trigger = assembler._detect_triggers("debug mode: why failing")
        assert trigger == "debug mode"
        assert "module-technical" in modules
        assert "module-analysis" in modules

    def test_detect_triggers_none(self, assembler):
        modules, trigger = assembler._detect_triggers("hello world")
        assert trigger is None
        assert modules == []

    def test_detect_keywords_technical(self, assembler):
        found = assembler._detect_keywords("fix the python bug")
        assert "module-technical" in found
        assert "module-analysis" in found

    def test_detect_keywords_creative(self, assembler):
        found = assembler._detect_keywords("design a new logo")
        assert "module-creative" in found

    def test_detect_keywords_research(self, assembler):
        found = assembler._detect_keywords("arxiv paper on llms")
        assert "module-research" in found

    def test_enforce_budget_minimal(self, assembler):
        names = list(assembler.TOKEN_ESTIMATES.keys())
        result = assembler._enforce_budget(names, 4000, "minimal")
        # Minimal caps at 1000 tokens
        total = sum(assembler.TOKEN_ESTIMATES.get(n, 300) for n in result)
        assert total <= 1000

    def test_enforce_budget_standard(self, assembler):
        names = list(assembler.TOKEN_ESTIMATES.keys())
        result = assembler._enforce_budget(names, 4000, "standard")
        total = sum(assembler.TOKEN_ESTIMATES.get(n, 300) for n in result)
        assert total <= 2000

    def test_enforce_budget_full(self, assembler):
        names = list(assembler.TOKEN_ESTIMATES.keys())
        result = assembler._enforce_budget(names, 4000, "full")
        total = sum(assembler.TOKEN_ESTIMATES.get(n, 300) for n in result)
        assert total <= 4500

    def test_assemble_returns_expected_keys(self, assembler):
        result = assembler.assemble("hello world", mode="minimal")
        assert "mode" in result
        assert "trigger" in result
        assert "modules_loaded" in result
        assert "modules_missing" in result
        assert "total_tokens" in result
        assert "assembled_block" in result
        assert result["mode"] == "minimal"

    def test_assemble_full_load(self, assembler):
        result = assembler.assemble("full load everything", max_tokens=5000)
        assert result["mode"] == "full"
        assert result["trigger"] == "full load"

    def test_token_estimates_cover_all_modules(self, assembler):
        # Every module in TOKEN_ESTIMATES should have a positive token count
        for name, tokens in assembler.TOKEN_ESTIMATES.items():
            assert tokens > 0
            assert isinstance(tokens, int)

    def test_mandatory_modules_subset_of_estimates(self, assembler):
        # All mandatory modules must have token estimates
        for name in assembler.MANDATORY_MODULES:
            assert name in assembler.TOKEN_ESTIMATES
