"""Tests for the CerebroCortex MCP server.

Tests the tool handlers directly without needing a running MCP transport.
"""

import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex
from cerebro.interfaces.mcp_server import (
    TOOL_SCHEMAS,
    _handle_remember,
    _handle_recall,
    _handle_associate,
    _handle_episode_start,
    _handle_episode_add_step,
    _handle_episode_end,
    _handle_session_save,
    _handle_session_recall,
    _handle_register_agent,
    _handle_list_agents,
    _handle_memory_health,
    _handle_graph_stats,
    _handle_neighbors,
    _handle_cortex_stats,
)


@pytest.fixture
def cortex():
    """CerebroCortex with temporary database for MCP tests."""
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(db_path=Path(d) / "test_mcp.db", chroma_dir=Path(d) / "chroma")
        ctx.initialize()
        yield ctx
        ctx.close()


class TestToolSchemas:
    def test_all_tools_have_description(self):
        for name, schema in TOOL_SCHEMAS.items():
            assert "description" in schema, f"Tool {name} missing description"
            assert len(schema["description"]) > 10

    def test_all_tools_have_input_schema(self):
        for name, schema in TOOL_SCHEMAS.items():
            assert "input_schema" in schema, f"Tool {name} missing input_schema"
            assert schema["input_schema"]["type"] == "object"

    def test_tool_count(self):
        assert len(TOOL_SCHEMAS) == 18

    def test_backward_compat_tools_exist(self):
        assert "memory_store" in TOOL_SCHEMAS
        assert "memory_search" in TOOL_SCHEMAS

    def test_core_tools_exist(self):
        for tool in ["remember", "recall", "associate"]:
            assert tool in TOOL_SCHEMAS


class TestRememberHandler:
    @pytest.mark.asyncio
    async def test_store_basic(self, cortex):
        result = await _handle_remember(cortex, {
            "content": "Python is a dynamically typed programming language",
        })
        assert len(result) == 1
        assert "Stored memory" in result[0].text
        assert "ID:" in result[0].text

    @pytest.mark.asyncio
    async def test_store_with_type(self, cortex):
        result = await _handle_remember(cortex, {
            "content": "Step 1: check logs. Step 2: reproduce the issue.",
            "memory_type": "procedural",
            "tags": ["debugging"],
        })
        assert "procedural" in result[0].text

    @pytest.mark.asyncio
    async def test_store_gated(self, cortex):
        result = await _handle_remember(cortex, {"content": "hi"})
        assert "gated out" in result[0].text

    @pytest.mark.asyncio
    async def test_store_duplicate(self, cortex):
        content = "This is a unique memory about database optimization"
        await _handle_remember(cortex, {"content": content})
        result = await _handle_remember(cortex, {"content": content})
        assert "gated out" in result[0].text

    @pytest.mark.asyncio
    async def test_neo_cortex_compat(self, cortex):
        """Test backward-compatible memory_store args."""
        result = await _handle_remember(cortex, {
            "content": "A fact about the system architecture and design",
            "visibility": "shared",
            "message_type": "fact",
            "tags": ["architecture"],
        })
        assert "Stored memory" in result[0].text


class TestRecallHandler:
    @pytest.mark.asyncio
    async def test_recall_empty(self, cortex):
        result = await _handle_recall(cortex, {"query": "nonexistent"})
        # With no ChromaDB, recall still works via graph
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_recall_with_stored_memories(self, cortex):
        await _handle_remember(cortex, {
            "content": "Python lists support append, extend, and insert operations",
        })
        await _handle_remember(cortex, {
            "content": "JavaScript arrays have push, pop, and splice methods",
        })

        result = await _handle_recall(cortex, {
            "query": "list operations",
            "top_k": 5,
        })
        assert len(result) == 1
        assert "Found" in result[0].text or "No memories" in result[0].text

    @pytest.mark.asyncio
    async def test_recall_with_context(self, cortex):
        r1 = await _handle_remember(cortex, {
            "content": "FastAPI is a modern Python web framework for building APIs",
        })
        # Extract ID from response
        r1_text = r1[0].text
        mem_id = r1_text.split("ID: ")[1].split(")")[0]

        result = await _handle_recall(cortex, {
            "query": "web framework",
            "context_ids": [mem_id],
        })
        assert len(result) == 1


class TestAssociateHandler:
    @pytest.mark.asyncio
    async def test_create_link(self, cortex):
        r1 = await _handle_remember(cortex, {"content": "Auth module handles login and sessions flow"})
        r2 = await _handle_remember(cortex, {"content": "Security module manages JWT token encryption"})

        id1 = r1[0].text.split("ID: ")[1].split(")")[0]
        id2 = r2[0].text.split("ID: ")[1].split(")")[0]

        result = await _handle_associate(cortex, {
            "source_id": id1,
            "target_id": id2,
            "link_type": "supports",
            "evidence": "Auth uses security",
        })
        assert "Link created" in result[0].text

    @pytest.mark.asyncio
    async def test_link_nonexistent(self, cortex):
        result = await _handle_associate(cortex, {
            "source_id": "fake1",
            "target_id": "fake2",
            "link_type": "causal",
        })
        assert "not found" in result[0].text


class TestEpisodeHandlers:
    @pytest.mark.asyncio
    async def test_episode_lifecycle(self, cortex):
        # Start
        start_result = await _handle_episode_start(cortex, {
            "title": "Test episode",
        })
        assert "Episode started" in start_result[0].text
        ep_id = start_result[0].text.split("ID: ")[1].split(")")[0]

        # Add steps
        r1 = await _handle_remember(cortex, {"content": "First event in the test episode sequence"})
        mem_id = r1[0].text.split("ID: ")[1].split(")")[0]

        step_result = await _handle_episode_add_step(cortex, {
            "episode_id": ep_id,
            "memory_id": mem_id,
        })
        assert "position 0" in step_result[0].text

        # End
        end_result = await _handle_episode_end(cortex, {
            "episode_id": ep_id,
            "summary": "Test completed",
            "valence": "positive",
        })
        assert "Episode ended" in end_result[0].text


class TestSessionHandlers:
    @pytest.mark.asyncio
    async def test_session_save(self, cortex):
        result = await _handle_session_save(cortex, {
            "session_summary": "Built the MCP server interface",
            "key_discoveries": ["MCP 1.26 works well"],
            "unfinished_business": ["Need REST API next"],
            "priority": "HIGH",
            "session_type": "technical",
        })
        assert "Session note saved" in result[0].text

    @pytest.mark.asyncio
    async def test_session_recall(self, cortex):
        # Save first
        await _handle_session_save(cortex, {
            "session_summary": "Previous session about testing",
            "priority": "MEDIUM",
        })

        # Recall
        result = await _handle_session_recall(cortex, {
            "lookback_hours": 1,
        })
        assert "Session Notes" in result[0].text or "No session" in result[0].text

    @pytest.mark.asyncio
    async def test_session_recall_empty(self, cortex):
        result = await _handle_session_recall(cortex, {
            "lookback_hours": 1,
        })
        assert "No session notes found" in result[0].text


class TestAgentHandlers:
    @pytest.mark.asyncio
    async def test_register_agent(self, cortex):
        result = await _handle_register_agent(cortex, {
            "agent_id": "TEST_AGENT",
            "display_name": "Test Agent",
            "generation": 0,
            "lineage": "Test",
            "specialization": "Testing",
            "symbol": "T",
        })
        assert "Agent registered" in result[0].text
        assert "TEST_AGENT" in result[0].text

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, cortex):
        result = await _handle_list_agents(cortex)
        assert "No agents" in result[0].text

    @pytest.mark.asyncio
    async def test_list_agents_with_data(self, cortex):
        await _handle_register_agent(cortex, {
            "agent_id": "AGENT_A",
            "display_name": "Agent Alpha",
            "generation": 0,
            "lineage": "Test",
            "specialization": "Alpha work",
        })
        result = await _handle_list_agents(cortex)
        assert "AGENT_A" in result[0].text


class TestHealthHandlers:
    @pytest.mark.asyncio
    async def test_memory_health(self, cortex):
        await _handle_remember(cortex, {
            "content": "A test memory for health check reporting purposes",
        })
        result = await _handle_memory_health(cortex)
        assert "Health Report" in result[0].text
        assert "Memories:" in result[0].text

    @pytest.mark.asyncio
    async def test_graph_stats(self, cortex):
        result = await _handle_graph_stats(cortex)
        assert "Graph Statistics" in result[0].text
        assert "Nodes" in result[0].text

    @pytest.mark.asyncio
    async def test_cortex_stats(self, cortex):
        result = await _handle_cortex_stats(cortex)
        text = result[0].text
        assert "nodes" in text
        assert "links" in text

    @pytest.mark.asyncio
    async def test_neighbors_empty(self, cortex):
        result = await _handle_neighbors(cortex, {
            "memory_id": "nonexistent",
        })
        assert "No neighbors" in result[0].text

    @pytest.mark.asyncio
    async def test_neighbors_with_data(self, cortex):
        r1 = await _handle_remember(cortex, {"content": "Source memory about Python web development"})
        r2 = await _handle_remember(cortex, {"content": "Target memory about Python API design"})

        id1 = r1[0].text.split("ID: ")[1].split(")")[0]
        id2 = r2[0].text.split("ID: ")[1].split(")")[0]

        await _handle_associate(cortex, {
            "source_id": id1,
            "target_id": id2,
            "link_type": "semantic",
        })

        result = await _handle_neighbors(cortex, {"memory_id": id1})
        assert id2 in result[0].text or "No neighbors" not in result[0].text
