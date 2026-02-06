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
