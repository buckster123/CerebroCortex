"""Tests for the CerebroCortex MCP server.

Tests the tool handlers directly without needing a running MCP transport.
"""

import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex
from cerebro.interfaces.mcp_server import (
    TOOL_SCHEMAS, list_resources, list_resource_templates, list_prompts,
    _handle_remember,
    _handle_recall,
    _handle_associate,
    _handle_get_memory,
    _handle_delete_memory,
    _handle_update_memory,
    _handle_share_memory,
    _handle_episode_start,
    _handle_episode_add_step,
    _handle_episode_end,
    _handle_list_episodes,
    _handle_get_episode,
    _handle_get_episode_memories,
    _handle_store_intention,
    _handle_list_intentions,
    _handle_resolve_intention,
    _handle_find_path,
    _handle_common_neighbors,
    _handle_create_schema,
    _handle_list_schemas,
    _handle_store_procedure,
    _handle_list_procedures,
    _handle_record_procedure_outcome,
    _handle_emotional_summary,
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
        assert len(TOOL_SCHEMAS) == 39

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


class TestGetMemoryHandler:
    @pytest.mark.asyncio
    async def test_get_memory(self, cortex):
        r = await _handle_remember(cortex, {
            "content": "Python is a dynamically typed programming language",
        })
        mem_id = r[0].text.split("ID: ")[1].split(")")[0]

        result = await _handle_get_memory(cortex, {"memory_id": mem_id})
        assert mem_id in result[0].text
        assert "Python" in result[0].text

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, cortex):
        result = await _handle_get_memory(cortex, {"memory_id": "nonexistent"})
        assert "not found" in result[0].text


class TestDeleteMemoryHandler:
    @pytest.mark.asyncio
    async def test_delete_memory(self, cortex):
        r = await _handle_remember(cortex, {
            "content": "A temporary memory about testing deletion flows",
        })
        mem_id = r[0].text.split("ID: ")[1].split(")")[0]

        result = await _handle_delete_memory(cortex, {"memory_id": mem_id})
        assert "Deleted" in result[0].text

        # Verify it's gone
        result2 = await _handle_get_memory(cortex, {"memory_id": mem_id})
        assert "not found" in result2[0].text

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, cortex):
        result = await _handle_delete_memory(cortex, {"memory_id": "nonexistent"})
        assert "not found" in result[0].text


class TestUpdateMemoryHandler:
    @pytest.mark.asyncio
    async def test_update_salience(self, cortex):
        r = await _handle_remember(cortex, {
            "content": "An important memory about system architecture patterns",
        })
        mem_id = r[0].text.split("ID: ")[1].split(")")[0]

        result = await _handle_update_memory(cortex, {
            "memory_id": mem_id,
            "salience": 0.95,
        })
        assert "Updated" in result[0].text
        assert "0.95" in result[0].text

    @pytest.mark.asyncio
    async def test_update_tags(self, cortex):
        r = await _handle_remember(cortex, {
            "content": "Memory about Python web frameworks and their ecosystems",
        })
        mem_id = r[0].text.split("ID: ")[1].split(")")[0]

        result = await _handle_update_memory(cortex, {
            "memory_id": mem_id,
            "tags": ["python", "web", "important"],
        })
        assert "Updated" in result[0].text
        assert "python" in result[0].text

    @pytest.mark.asyncio
    async def test_update_content(self, cortex):
        r = await _handle_remember(cortex, {
            "content": "Original content about testing the update flow here",
        })
        mem_id = r[0].text.split("ID: ")[1].split(")")[0]

        result = await _handle_update_memory(cortex, {
            "memory_id": mem_id,
            "content": "Updated content about the new testing flow patterns",
        })
        assert "Updated" in result[0].text

        # Verify content changed
        get_result = await _handle_get_memory(cortex, {"memory_id": mem_id})
        assert "Updated content" in get_result[0].text

    @pytest.mark.asyncio
    async def test_update_not_found(self, cortex):
        result = await _handle_update_memory(cortex, {
            "memory_id": "nonexistent",
            "salience": 0.5,
        })
        assert "not found" in result[0].text


class TestEpisodeQueryHandlers:
    @pytest.mark.asyncio
    async def test_list_episodes_empty(self, cortex):
        result = await _handle_list_episodes(cortex, {})
        assert "No episodes" in result[0].text

    @pytest.mark.asyncio
    async def test_list_episodes_with_data(self, cortex):
        start = await _handle_episode_start(cortex, {"title": "Test ep"})
        ep_id = start[0].text.split("ID: ")[1].split(")")[0]
        await _handle_episode_end(cortex, {"episode_id": ep_id})

        result = await _handle_list_episodes(cortex, {"limit": 5})
        assert "Test ep" in result[0].text or ep_id in result[0].text

    @pytest.mark.asyncio
    async def test_get_episode(self, cortex):
        start = await _handle_episode_start(cortex, {"title": "Detail ep"})
        ep_id = start[0].text.split("ID: ")[1].split(")")[0]

        result = await _handle_get_episode(cortex, {"episode_id": ep_id})
        assert ep_id in result[0].text

    @pytest.mark.asyncio
    async def test_get_episode_not_found(self, cortex):
        result = await _handle_get_episode(cortex, {"episode_id": "nonexistent"})
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_get_episode_memories(self, cortex):
        start = await _handle_episode_start(cortex, {"title": "Mem ep"})
        ep_id = start[0].text.split("ID: ")[1].split(")")[0]
        r = await _handle_remember(cortex, {"content": "Memory in episode for testing purposes"})
        mem_id = r[0].text.split("ID: ")[1].split(")")[0]
        await _handle_episode_add_step(cortex, {"episode_id": ep_id, "memory_id": mem_id})

        result = await _handle_get_episode_memories(cortex, {"episode_id": ep_id})
        assert "Memory in episode" in result[0].text


class TestIntentionHandlers:
    @pytest.mark.asyncio
    async def test_store_intention(self, cortex):
        result = await _handle_store_intention(cortex, {
            "content": "Remember to update the documentation after refactoring",
        })
        assert "Stored intention" in result[0].text
        assert "ID:" in result[0].text

    @pytest.mark.asyncio
    async def test_list_intentions_empty(self, cortex):
        result = await _handle_list_intentions(cortex, {})
        assert "No pending" in result[0].text

    @pytest.mark.asyncio
    async def test_list_intentions_with_data(self, cortex):
        await _handle_store_intention(cortex, {
            "content": "TODO: write more tests for the memory system",
        })
        result = await _handle_list_intentions(cortex, {})
        assert "write more tests" in result[0].text

    @pytest.mark.asyncio
    async def test_resolve_intention(self, cortex):
        r = await _handle_store_intention(cortex, {
            "content": "Intention to resolve after completing the task",
        })
        mem_id = r[0].text.split("ID: ")[1].split(")")[0]

        result = await _handle_resolve_intention(cortex, {"memory_id": mem_id})
        assert "Resolved" in result[0].text

    @pytest.mark.asyncio
    async def test_resolve_intention_not_found(self, cortex):
        result = await _handle_resolve_intention(cortex, {"memory_id": "nonexistent"})
        assert "not found" in result[0].text


class TestGraphExplorationHandlers:
    @pytest.mark.asyncio
    async def test_find_path(self, cortex):
        r1 = await _handle_remember(cortex, {"content": "Source node for path finding test"})
        r2 = await _handle_remember(cortex, {"content": "Target node for path finding test"})
        id1 = r1[0].text.split("ID: ")[1].split(")")[0]
        id2 = r2[0].text.split("ID: ")[1].split(")")[0]
        await _handle_associate(cortex, {"source_id": id1, "target_id": id2, "link_type": "semantic"})

        result = await _handle_find_path(cortex, {"source_id": id1, "target_id": id2})
        assert "Path" in result[0].text or id1 in result[0].text

    @pytest.mark.asyncio
    async def test_find_path_no_path(self, cortex):
        result = await _handle_find_path(cortex, {"source_id": "fake1", "target_id": "fake2"})
        assert "No path" in result[0].text

    @pytest.mark.asyncio
    async def test_common_neighbors(self, cortex):
        result = await _handle_common_neighbors(cortex, {"id_a": "fake1", "id_b": "fake2"})
        assert "No common" in result[0].text or "common" in result[0].text.lower()


class TestSchemaHandlers:
    @pytest.mark.asyncio
    async def test_create_schema(self, cortex):
        r1 = await _handle_remember(cortex, {"content": "Python uses indentation for code blocks"})
        r2 = await _handle_remember(cortex, {"content": "JavaScript uses curly braces for code blocks"})
        id1 = r1[0].text.split("ID: ")[1].split(")")[0]
        id2 = r2[0].text.split("ID: ")[1].split(")")[0]

        result = await _handle_create_schema(cortex, {
            "content": "Programming languages use different block delimiters",
            "source_ids": [id1, id2],
            "tags": ["programming"],
        })
        assert "Created schema" in result[0].text

    @pytest.mark.asyncio
    async def test_list_schemas_empty(self, cortex):
        result = await _handle_list_schemas(cortex, {})
        assert "No schemas" in result[0].text


class TestProcedureHandlers:
    @pytest.mark.asyncio
    async def test_store_procedure(self, cortex):
        result = await _handle_store_procedure(cortex, {
            "content": "Step 1: Read logs. Step 2: Reproduce. Step 3: Fix.",
            "tags": ["debugging"],
        })
        assert "Stored procedure" in result[0].text

    @pytest.mark.asyncio
    async def test_list_procedures_empty(self, cortex):
        result = await _handle_list_procedures(cortex, {})
        assert "No procedures" in result[0].text

    @pytest.mark.asyncio
    async def test_record_outcome(self, cortex):
        r = await _handle_store_procedure(cortex, {
            "content": "Strategy for handling API timeout errors gracefully",
        })
        proc_id = r[0].text.split("ID: ")[1].split(")")[0]
        result = await _handle_record_procedure_outcome(cortex, {
            "procedure_id": proc_id, "success": True,
        })
        assert "Recorded" in result[0].text


class TestEmotionalSummaryHandler:
    @pytest.mark.asyncio
    async def test_emotional_summary(self, cortex):
        await _handle_remember(cortex, {"content": "A test memory for emotional summary reporting"})
        result = await _handle_emotional_summary(cortex, {})
        assert len(result) == 1


class TestShareMemoryHandler:
    @pytest.mark.asyncio
    async def test_share_memory_tool_exists(self):
        assert "share_memory" in TOOL_SCHEMAS

    @pytest.mark.asyncio
    async def test_share_memory_owner_can_share(self, cortex):
        # Store a private memory as ALICE
        result = await _handle_remember(cortex, {
            "content": "Alice private memory for share test",
            "agent_id": "ALICE",
            "visibility": "private",
        })
        mem_id = result[0].text.split("ID: ")[1].split(")")[0]

        # ALICE shares it
        result = await _handle_share_memory(cortex, {
            "memory_id": mem_id,
            "visibility": "shared",
            "agent_id": "ALICE",
        })
        assert "Visibility changed" in result[0].text

    @pytest.mark.asyncio
    async def test_share_memory_non_owner_rejected(self, cortex):
        # Store a private memory as ALICE
        result = await _handle_remember(cortex, {
            "content": "Alice private memory for share rejection test",
            "agent_id": "ALICE",
            "visibility": "private",
        })
        mem_id = result[0].text.split("ID: ")[1].split(")")[0]

        # BOB tries to share it
        result = await _handle_share_memory(cortex, {
            "memory_id": mem_id,
            "visibility": "shared",
            "agent_id": "BOB",
        })
        assert "not authorized" in result[0].text.lower() or "Not found" in result[0].text


class TestMCPResources:
    @pytest.mark.asyncio
    async def test_list_resources(self):
        resources = await list_resources()
        assert len(resources) >= 1
        uris = [str(r.uri) for r in resources]
        assert "cerebro://stats" in uris

    @pytest.mark.asyncio
    async def test_list_resource_templates(self):
        templates = await list_resource_templates()
        assert len(templates) >= 2


class TestMCPPrompts:
    @pytest.mark.asyncio
    async def test_list_prompts(self):
        prompts = await list_prompts()
        assert len(prompts) == 3
        names = [p.name for p in prompts]
        assert "session_handoff" in names
        assert "memory_review" in names
        assert "context_briefing" in names
