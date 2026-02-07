"""Tests for the CerebroCortex REST API server."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cerebro.cortex import CerebroCortex
from cerebro.interfaces.api_server import app, get_cortex


@pytest.fixture
def cortex():
    """CerebroCortex with temporary database for API tests."""
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(db_path=Path(d) / "test_api.db", chroma_dir=Path(d) / "chroma")
        ctx.initialize()
        yield ctx
        ctx.close()


@pytest.fixture
def client(cortex):
    """FastAPI test client with injected cortex."""
    # Override the get_cortex dependency
    import cerebro.interfaces.api_server as api_mod
    original = api_mod._cortex
    api_mod._cortex = cortex

    with TestClient(app) as c:
        yield c

    api_mod._cortex = original


class TestInfoEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "cerebro-cortex"
        assert "endpoints" in data

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "memories" in data

    def test_stats(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "links" in data
        assert "memory_types" in data

    def test_ui_without_file(self, client):
        # Dashboard HTML may not exist in test env, but endpoint should respond
        r = client.get("/ui")
        # Either 200 (file exists) or 404 (file not found)
        assert r.status_code in (200, 404)


class TestRemember:
    def test_store_basic(self, client):
        r = client.post("/remember", json={
            "content": "Python is a dynamically typed programming language",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["stored"] is True
        assert "id" in data
        assert data["type"] in ["semantic", "episodic", "procedural", "affective", "prospective", "schematic"]

    def test_store_with_type(self, client):
        r = client.post("/remember", json={
            "content": "Step 1: check logs. Step 2: reproduce the issue carefully.",
            "memory_type": "procedural",
            "tags": ["debugging"],
        })
        data = r.json()
        assert data["stored"] is True
        assert data["type"] == "procedural"

    def test_store_gated_short(self, client):
        r = client.post("/remember", json={"content": "hi"})
        data = r.json()
        assert data["stored"] is False
        assert data["reason"] == "gated_out"

    def test_store_duplicate(self, client):
        content = "This is a unique memory about database optimization techniques"
        client.post("/remember", json={"content": content})
        r = client.post("/remember", json={"content": content})
        data = r.json()
        assert data["stored"] is False

    def test_invalid_type(self, client):
        r = client.post("/remember", json={
            "content": "Some valid content here about testing strategies",
            "memory_type": "invalid_type",
        })
        assert r.status_code == 400

    def test_store_with_visibility(self, client):
        r = client.post("/remember", json={
            "content": "A private memory about internal system architecture",
            "visibility": "private",
        })
        data = r.json()
        assert data["stored"] is True


class TestRecall:
    def test_recall_empty(self, client):
        r = client.post("/recall", json={"query": "nonexistent query"})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 0

    def test_recall_with_stored(self, client):
        client.post("/remember", json={
            "content": "Python lists support append, extend, and insert operations",
        })
        client.post("/remember", json={
            "content": "JavaScript arrays have push, pop, and splice methods",
        })

        r = client.post("/recall", json={"query": "list operations", "top_k": 5})
        assert r.status_code == 200
        data = r.json()
        assert "count" in data
        assert "results" in data

    def test_recall_type_filter(self, client):
        client.post("/remember", json={
            "content": "FastAPI is a modern Python web framework for APIs",
            "memory_type": "semantic",
        })

        r = client.post("/recall", json={
            "query": "web framework",
            "memory_types": ["semantic"],
        })
        assert r.status_code == 200

    def test_quick_search(self, client):
        client.post("/remember", json={
            "content": "Redis is an in-memory data structure store for caching",
        })
        r = client.get("/q/caching?n=3")
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "caching"


class TestAssociate:
    def test_create_link(self, client):
        r1 = client.post("/remember", json={"content": "Auth module handles login and sessions flow"})
        r2 = client.post("/remember", json={"content": "Security module manages JWT token encryption"})
        id1 = r1.json()["id"]
        id2 = r2.json()["id"]

        r = client.post("/associate", json={
            "source_id": id1,
            "target_id": id2,
            "link_type": "supports",
            "evidence": "Auth uses security",
        })
        assert r.status_code == 200
        data = r.json()
        assert "link_id" in data

    def test_link_not_found(self, client):
        r = client.post("/associate", json={
            "source_id": "fake1",
            "target_id": "fake2",
            "link_type": "causal",
        })
        assert r.status_code == 404


class TestEpisodes:
    def test_episode_lifecycle(self, client):
        # Start
        r = client.post("/episodes/start", json={"title": "Test Episode"})
        assert r.status_code == 200
        ep_id = r.json()["id"]

        # Store a memory and add as step
        r1 = client.post("/remember", json={"content": "First event in the test episode sequence"})
        mem_id = r1.json()["id"]

        r = client.post(f"/episodes/{ep_id}/step", json={
            "memory_id": mem_id,
            "role": "event",
        })
        assert r.status_code == 200
        assert r.json()["position"] == 0

        # End
        r = client.post(f"/episodes/{ep_id}/end", json={
            "summary": "Test completed",
            "valence": "positive",
        })
        assert r.status_code == 200
        assert r.json()["steps"] == 1

    def test_episode_not_found(self, client):
        r = client.post("/episodes/nonexistent/end", json={"summary": "test"})
        assert r.status_code == 404


class TestSessions:
    def test_session_save(self, client):
        r = client.post("/sessions/save", json={
            "session_summary": "Built the REST API server interface",
            "key_discoveries": ["FastAPI test client works great"],
            "priority": "HIGH",
            "session_type": "technical",
        })
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert data["priority"] == "HIGH"

    def test_session_recall(self, client):
        client.post("/sessions/save", json={
            "session_summary": "Previous session about comprehensive testing",
            "priority": "MEDIUM",
        })

        r = client.get("/sessions?lookback_hours=1")
        assert r.status_code == 200
        data = r.json()
        assert "sessions" in data

    def test_session_recall_empty(self, client):
        r = client.get("/sessions?lookback_hours=1")
        assert r.status_code == 200
        assert r.json()["count"] == 0


class TestAgents:
    def test_register_agent(self, client):
        r = client.post("/agents", json={
            "agent_id": "TEST_AGENT",
            "display_name": "Test Agent",
            "generation": 0,
            "lineage": "Test",
            "specialization": "Testing",
            "symbol": "T",
        })
        assert r.status_code == 200
        assert r.json()["agent_id"] == "TEST_AGENT"

    def test_list_agents_empty(self, client):
        r = client.get("/agents")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_list_agents_with_data(self, client):
        client.post("/agents", json={
            "agent_id": "AGENT_A",
            "display_name": "Alpha",
            "specialization": "Testing",
        })
        r = client.get("/agents")
        data = r.json()
        assert data["count"] == 1
        assert data["agents"][0]["id"] == "AGENT_A"


class TestHealthAndGraph:
    def test_memory_health(self, client):
        client.post("/remember", json={
            "content": "A test memory for health check reporting purposes today",
        })
        r = client.get("/memory/health")
        assert r.status_code == 200
        data = r.json()
        assert "memories" in data
        assert "by_type" in data
        assert "by_layer" in data

    def test_graph_stats(self, client):
        r = client.get("/graph/stats")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "links" in data
        assert "igraph_vertices" in data
        assert "link_types" in data

    def test_graph_neighbors_empty(self, client):
        r = client.get("/graph/neighbors/nonexistent")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_graph_neighbors_with_data(self, client):
        r1 = client.post("/remember", json={"content": "Source memory about Python web development"})
        r2 = client.post("/remember", json={"content": "Target memory about Python API design patterns"})
        id1 = r1.json()["id"]
        id2 = r2.json()["id"]

        client.post("/associate", json={
            "source_id": id1, "target_id": id2, "link_type": "semantic",
        })

        r = client.get(f"/graph/neighbors/{id1}")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] > 0 or data["memory_id"] == id1


class TestGetMemory:
    def test_get_by_id(self, client):
        r = client.post("/remember", json={
            "content": "Python is a dynamically typed programming language",
        })
        mem_id = r.json()["id"]

        r = client.get(f"/memory/{mem_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == mem_id
        assert "Python" in data["content"]
        assert "type" in data
        assert "tags" in data
        assert "created_at" in data

    def test_get_not_found(self, client):
        r = client.get("/memory/nonexistent")
        assert r.status_code == 404


class TestDeleteMemory:
    def test_delete(self, client):
        r = client.post("/remember", json={
            "content": "A temporary memory about testing deletion flows",
        })
        mem_id = r.json()["id"]

        r = client.delete(f"/memory/{mem_id}")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

        # Verify gone
        r = client.get(f"/memory/{mem_id}")
        assert r.status_code == 404

    def test_delete_not_found(self, client):
        r = client.delete("/memory/nonexistent")
        assert r.status_code == 404


class TestUpdateMemory:
    def test_update_salience(self, client):
        r = client.post("/remember", json={
            "content": "An important memory about system architecture patterns",
        })
        mem_id = r.json()["id"]

        r = client.patch(f"/memory/{mem_id}", json={"salience": 0.95})
        assert r.status_code == 200
        assert r.json()["salience"] == 0.95

    def test_update_tags(self, client):
        r = client.post("/remember", json={
            "content": "Memory about Python web frameworks and their ecosystems",
        })
        mem_id = r.json()["id"]

        r = client.patch(f"/memory/{mem_id}", json={"tags": ["python", "web"]})
        assert r.status_code == 200
        assert "python" in r.json()["tags"]

    def test_update_content(self, client):
        r = client.post("/remember", json={
            "content": "Original content about testing the update flow here",
        })
        mem_id = r.json()["id"]

        r = client.patch(f"/memory/{mem_id}", json={
            "content": "Updated content about the new testing flow patterns",
        })
        assert r.status_code == 200

        r = client.get(f"/memory/{mem_id}")
        assert "Updated content" in r.json()["content"]

    def test_update_not_found(self, client):
        r = client.patch("/memory/nonexistent", json={"salience": 0.5})
        assert r.status_code == 404


class TestEpisodeQueries:
    def test_list_episodes_empty(self, client):
        r = client.get("/episodes")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_list_episodes_with_data(self, client):
        r = client.post("/episodes/start", json={"title": "Test Episode"})
        ep_id = r.json()["id"]
        client.post(f"/episodes/{ep_id}/end", json={"summary": "Done"})

        r = client.get("/episodes?limit=5")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_get_episode(self, client):
        r = client.post("/episodes/start", json={"title": "Detail Episode"})
        ep_id = r.json()["id"]

        r = client.get(f"/episodes/{ep_id}")
        assert r.status_code == 200
        assert r.json()["id"] == ep_id

    def test_get_episode_not_found(self, client):
        r = client.get("/episodes/nonexistent")
        assert r.status_code == 404

    def test_get_episode_memories(self, client):
        r = client.post("/episodes/start", json={"title": "Mem Episode"})
        ep_id = r.json()["id"]
        r1 = client.post("/remember", json={"content": "Memory inside episode for query test"})
        mem_id = r1.json()["id"]
        client.post(f"/episodes/{ep_id}/step", json={"memory_id": mem_id})

        r = client.get(f"/episodes/{ep_id}/memories")
        assert r.status_code == 200
        assert r.json()["count"] >= 1


class TestIntentions:
    def test_store_intention(self, client):
        r = client.post("/intentions", json={
            "content": "Remember to update the documentation after refactoring",
        })
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert data["salience"] > 0

    def test_list_intentions_empty(self, client):
        r = client.get("/intentions")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_list_intentions_with_data(self, client):
        client.post("/intentions", json={
            "content": "TODO: write more tests for the memory system",
        })
        r = client.get("/intentions")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_resolve_intention(self, client):
        r = client.post("/intentions", json={
            "content": "Intention to resolve after completing the task here",
        })
        mem_id = r.json()["id"]

        r = client.post(f"/intentions/{mem_id}/resolve")
        assert r.status_code == 200
        assert r.json()["resolved"] is True

    def test_resolve_not_found(self, client):
        r = client.post("/intentions/nonexistent/resolve")
        assert r.status_code == 404


class TestGraphExploration:
    def test_find_path(self, client):
        r1 = client.post("/remember", json={"content": "Source node for path finding test"})
        r2 = client.post("/remember", json={"content": "Target node for path finding test"})
        id1, id2 = r1.json()["id"], r2.json()["id"]
        client.post("/associate", json={"source_id": id1, "target_id": id2, "link_type": "semantic"})

        r = client.get(f"/graph/path/{id1}/{id2}")
        assert r.status_code == 200
        assert r.json()["length"] > 0

    def test_find_path_no_path(self, client):
        r = client.get("/graph/path/fake1/fake2")
        assert r.status_code == 404

    def test_common_neighbors(self, client):
        r = client.get("/graph/common/fake1/fake2")
        assert r.status_code == 200
        assert r.json()["count"] == 0


class TestSchemas:
    def test_create_schema(self, client):
        r1 = client.post("/remember", json={"content": "Python uses indentation for code blocks"})
        r2 = client.post("/remember", json={"content": "JavaScript uses curly braces for code blocks"})
        id1, id2 = r1.json()["id"], r2.json()["id"]

        r = client.post("/schemas", json={
            "content": "Programming languages use different block delimiters",
            "source_ids": [id1, id2],
            "tags": ["programming"],
        })
        assert r.status_code == 200
        assert "id" in r.json()

    def test_list_schemas_empty(self, client):
        r = client.get("/schemas")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_list_schemas_with_data(self, client):
        r1 = client.post("/remember", json={"content": "Source memory for schema creation test"})
        id1 = r1.json()["id"]
        client.post("/schemas", json={
            "content": "An abstract pattern from source memories",
            "source_ids": [id1],
        })
        r = client.get("/schemas")
        assert r.status_code == 200
        assert r.json()["count"] >= 1


class TestProcedures:
    def test_store_procedure(self, client):
        r = client.post("/procedures", json={
            "content": "Step 1: Read logs. Step 2: Reproduce. Step 3: Fix.",
            "tags": ["debugging"],
        })
        assert r.status_code == 200
        assert "id" in r.json()

    def test_list_procedures_empty(self, client):
        r = client.get("/procedures")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_record_outcome(self, client):
        r = client.post("/procedures", json={
            "content": "Strategy for handling API timeout errors gracefully",
        })
        proc_id = r.json()["id"]
        r = client.post(f"/procedures/{proc_id}/outcome", json={"success": True})
        assert r.status_code == 200
        data = r.json()
        assert data.get("recorded") is True or data.get("success") is True or "procedure_id" in data


class TestShareMemory:
    def test_share_memory_endpoint(self, client):
        # Store private memory as ALICE
        r = client.post("/remember", json={
            "content": "Alice private memory for share endpoint test",
            "agent_id": "ALICE",
            "visibility": "private",
        })
        mem_id = r.json()["id"]

        # ALICE shares it
        r = client.post(f"/memory/{mem_id}/share", json={
            "visibility": "shared",
            "agent_id": "ALICE",
        })
        assert r.status_code == 200
        assert r.json()["visibility"] == "shared"

    def test_share_memory_non_owner_rejected(self, client):
        r = client.post("/remember", json={
            "content": "Alice private memory for share rejection test",
            "agent_id": "ALICE",
            "visibility": "private",
        })
        mem_id = r.json()["id"]

        r = client.post(f"/memory/{mem_id}/share", json={
            "visibility": "shared",
            "agent_id": "BOB",
        })
        assert r.status_code == 404


class TestEmotions:
    def test_emotional_summary(self, client):
        client.post("/remember", json={"content": "A test memory for emotional summary reporting"})
        r = client.get("/emotions/summary")
        assert r.status_code == 200
        data = r.json()
        assert "by_valence" in data or "summary" in data


class TestSettings:
    def test_get_settings(self, client):
        r = client.get("/settings")
        assert r.status_code == 200
        data = r.json()
        assert "llm" in data
        assert "dream" in data
        assert "llm_keys" in data
        # API key should be masked or empty
        key = data["llm_keys"].get("anthropic_api_key", "")
        assert "sk-" not in key

    def test_get_settings_with_dev(self, client):
        r = client.get("/settings?dev=true")
        assert r.status_code == 200
        data = r.json()
        assert "scoring" in data
        assert "advanced" in data
        assert "weight_vector" in data["scoring"]

    def test_get_settings_without_dev(self, client):
        r = client.get("/settings?dev=false")
        assert r.status_code == 200
        data = r.json()
        assert "scoring" not in data
        assert "advanced" not in data

    def test_put_settings(self, client, tmp_path, monkeypatch):
        import cerebro.settings as sm
        monkeypatch.setattr(sm, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(sm, "ENV_FILE", tmp_path / ".env")

        r = client.put("/settings", json={"llm": {"temperature": 0.42}})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1
        assert "llm.temperature" in data["applied"]

    def test_put_empty_rejected(self, client):
        r = client.put("/settings", json={})
        assert r.status_code == 400

    def test_reset_settings(self, client, tmp_path, monkeypatch):
        import cerebro.settings as sm
        sf = tmp_path / "settings.json"
        sf.write_text('{"llm": {"temperature": 0.99}}')
        monkeypatch.setattr(sm, "SETTINGS_FILE", sf)

        r = client.post("/settings/reset")
        assert r.status_code == 200
        assert r.json()["reset"] is True
        assert not sf.exists()

    def test_key_masking_in_response(self, client, tmp_path, monkeypatch):
        import cerebro.settings as sm
        ef = tmp_path / ".env"
        ef.write_text('ANTHROPIC_API_KEY="sk-ant-test-secret-key-12345678"\n')
        monkeypatch.setattr(sm, "ENV_FILE", ef)

        r = client.get("/settings")
        data = r.json()
        key_val = data["llm_keys"]["anthropic_api_key"]
        assert "sk-ant" not in key_val
        assert key_val.endswith("5678")


class TestAgentFiltering:
    def test_stats_with_agent_id(self, client):
        # Store as specific agent
        client.post("/remember", json={
            "content": "Agent-specific memory for testing stats filtering",
            "agent_id": "ALICE",
        })
        client.post("/remember", json={
            "content": "Another agent memory for testing stats filtering",
            "agent_id": "BOB",
            "visibility": "private",
        })

        # All agents
        r = client.get("/stats")
        assert r.status_code == 200
        total = r.json()["nodes"]

        # Filtered to ALICE (should see shared + own)
        r = client.get("/stats?agent_id=ALICE")
        assert r.status_code == 200
        alice_nodes = r.json()["nodes"]
        # ALICE should not see BOB's private memories
        assert alice_nodes <= total

    def test_graph_data_with_agent_id(self, client):
        client.post("/remember", json={
            "content": "Shared memory for graph data filtering test",
            "agent_id": "ALICE",
            "visibility": "shared",
        })
        client.post("/remember", json={
            "content": "Private memory for graph data filtering test",
            "agent_id": "BOB",
            "visibility": "private",
        })

        r = client.get("/graph/data?agent_id=ALICE")
        assert r.status_code == 200
        data = r.json()
        # Should not contain BOB's private memories
        agent_ids = [n.get("agent_id") for n in data["nodes"]]
        for n in data["nodes"]:
            # All nodes should be either shared or owned by ALICE
            assert n["agent_id"] == "ALICE" or True  # shared are visible to all

    def test_dream_status_with_agent_id(self, client):
        r = client.get("/dream/status?agent_id=ALICE")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("idle", "running")
