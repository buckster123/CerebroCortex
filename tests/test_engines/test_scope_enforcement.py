"""Tests for multi-agent scope enforcement.

Verifies that visibility rules (SHARED/PRIVATE/THREAD) are enforced
across all CerebroCortex operations including recall, spreading activation,
engine queries, and CRUD access checks.
"""

import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex, _scope_sql
from cerebro.models.memory import MemoryMetadata, MemoryNode
from cerebro.types import LinkType, MemoryType, Visibility


# =============================================================================
# TestCanAccess - Unit tests for the _can_access static method
# =============================================================================

class TestCanAccess:
    """Test the _can_access visibility gate."""

    def test_shared_accessible_to_all(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        node = multi_agent_cortex["alice_shared"]
        assert CerebroCortex._can_access(node, agent_id="ALICE")
        assert CerebroCortex._can_access(node, agent_id="BOB")
        assert CerebroCortex._can_access(node, agent_id="CHARLIE")

    def test_private_owner_only(self, multi_agent_cortex):
        node = multi_agent_cortex["alice_private"]
        assert CerebroCortex._can_access(node, agent_id="ALICE")
        assert not CerebroCortex._can_access(node, agent_id="BOB")
        assert not CerebroCortex._can_access(node, agent_id="CHARLIE")

    def test_private_bob_owner_only(self, multi_agent_cortex):
        node = multi_agent_cortex["bob_private"]
        assert CerebroCortex._can_access(node, agent_id="BOB")
        assert not CerebroCortex._can_access(node, agent_id="ALICE")

    def test_thread_with_matching_thread(self, multi_agent_cortex):
        node = multi_agent_cortex["thread_node"]
        # Anyone with matching thread can access
        assert CerebroCortex._can_access(node, agent_id="BOB", conversation_thread="thread-42")
        assert CerebroCortex._can_access(node, agent_id="ALICE", conversation_thread="thread-42")

    def test_thread_without_matching_thread(self, multi_agent_cortex):
        node = multi_agent_cortex["thread_node"]
        # Without matching thread, only owner can access
        assert CerebroCortex._can_access(node, agent_id="ALICE")
        assert not CerebroCortex._can_access(node, agent_id="BOB")
        # Wrong thread
        assert not CerebroCortex._can_access(node, agent_id="BOB", conversation_thread="thread-99")

    def test_no_filter_returns_all(self, multi_agent_cortex):
        """No agent_id = see everything (backwards compat)."""
        for key in ["alice_shared", "alice_private", "bob_private", "thread_node"]:
            node = multi_agent_cortex[key]
            assert CerebroCortex._can_access(node, agent_id=None)


# =============================================================================
# TestRecallScope - Scope enforcement in recall pipeline
# =============================================================================

class TestRecallScope:
    """Test that recall respects visibility scope."""

    def test_bob_sees_alice_shared(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        results = ctx.recall("Python decorators metaprogramming", agent_id="BOB")
        result_ids = {node.id for node, _ in results}
        assert multi_agent_cortex["alice_shared"].id in result_ids

    def test_bob_cannot_see_alice_private(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        results = ctx.recall("secret configuration API keys", agent_id="BOB")
        result_ids = {node.id for node, _ in results}
        assert multi_agent_cortex["alice_private"].id not in result_ids

    def test_alice_sees_own_private(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        results = ctx.recall("secret configuration API keys", agent_id="ALICE")
        result_ids = {node.id for node, _ in results}
        assert multi_agent_cortex["alice_private"].id in result_ids

    def test_no_filter_returns_everything(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        # No agent_id = backwards compatible, see everything
        results = ctx.recall("Python decorators secret debugging deployment", agent_id=None)
        result_ids = {node.id for node, _ in results}
        # Should be able to see all memories including private ones
        assert multi_agent_cortex["alice_shared"].id in result_ids

    def test_thread_visibility_with_matching_thread(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        results = ctx.recall(
            "deployment strategy",
            agent_id="BOB",
            conversation_thread="thread-42",
        )
        result_ids = {node.id for node, _ in results}
        assert multi_agent_cortex["thread_node"].id in result_ids

    def test_thread_visibility_without_matching_thread(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        results = ctx.recall(
            "deployment strategy",
            agent_id="BOB",
        )
        result_ids = {node.id for node, _ in results}
        assert multi_agent_cortex["thread_node"].id not in result_ids


# =============================================================================
# TestSpreadingScope - Spreading activation respects scope
# =============================================================================

class TestSpreadingScope:
    """Test that spreading activation doesn't leak across boundaries."""

    def test_activation_stays_within_shared(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        alice_shared = multi_agent_cortex["alice_shared"]

        # Create a link from alice_shared to alice_private
        alice_private = multi_agent_cortex["alice_private"]
        ctx.links.create_link(
            alice_shared.id, alice_private.id, LinkType.SEMANTIC,
            weight=0.9, source="test",
        )

        # Spread from alice_shared as BOB - should NOT reach alice_private
        activated = ctx.links.spread_activation(
            seed_ids=[alice_shared.id],
            seed_weights=[1.0],
            agent_id="BOB",
        )
        assert alice_private.id not in activated

    def test_activation_reaches_own_private(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        alice_shared = multi_agent_cortex["alice_shared"]
        alice_private = multi_agent_cortex["alice_private"]

        # Create a link from alice_shared to alice_private
        ctx.links.create_link(
            alice_shared.id, alice_private.id, LinkType.SEMANTIC,
            weight=0.9, source="test",
        )

        # Spread from alice_shared as ALICE - SHOULD reach alice_private
        activated = ctx.links.spread_activation(
            seed_ids=[alice_shared.id],
            seed_weights=[1.0],
            agent_id="ALICE",
        )
        assert alice_private.id in activated

    def test_activation_no_filter_reaches_all(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        alice_shared = multi_agent_cortex["alice_shared"]
        alice_private = multi_agent_cortex["alice_private"]

        ctx.links.create_link(
            alice_shared.id, alice_private.id, LinkType.SEMANTIC,
            weight=0.9, source="test",
        )

        # No agent_id = no filter
        activated = ctx.links.spread_activation(
            seed_ids=[alice_shared.id],
            seed_weights=[1.0],
        )
        assert alice_private.id in activated


# =============================================================================
# TestEngineScope - Engine queries filter by scope
# =============================================================================

class TestEngineScope:
    """Test that engine SQL queries filter by visibility."""

    def test_schemas_filtered_by_scope(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]

        # Create schemas with different visibility
        alice_schema = ctx.create_schema(
            content="Pattern: Python projects use virtual environments",
            source_ids=[multi_agent_cortex["alice_shared"].id],
            tags=["pattern"],
            agent_id="ALICE",
        )
        # Make the schema private via update
        ctx.update_memory(alice_schema.id, visibility=Visibility.PRIVATE)

        # BOB should not see private schemas
        schemas = ctx.list_schemas(agent_id="BOB")
        schema_ids = {s.id for s in schemas}
        assert alice_schema.id not in schema_ids

        # ALICE should see own private schema
        schemas = ctx.list_schemas(agent_id="ALICE")
        schema_ids = {s.id for s in schemas}
        assert alice_schema.id in schema_ids

    def test_procedures_filtered_by_scope(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]

        # Create a private procedure
        proc = ctx.store_procedure(
            content="Secret debugging workflow for internal systems",
            tags=["debugging"],
            agent_id="ALICE",
        )
        ctx.update_memory(proc.id, visibility=Visibility.PRIVATE)

        # BOB should not see it
        procs = ctx.list_procedures(agent_id="BOB")
        proc_ids = {p.id for p in procs}
        assert proc.id not in proc_ids

        # ALICE should see it
        procs = ctx.list_procedures(agent_id="ALICE")
        proc_ids = {p.id for p in procs}
        assert proc.id in proc_ids

    def test_intentions_filtered_by_scope(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]

        # Store a private intention
        intention = ctx.store_intention(
            content="Alice TODO: fix internal module",
            tags=["todo"],
            agent_id="ALICE",
        )
        ctx.update_memory(intention.id, visibility=Visibility.PRIVATE)

        # BOB should not see it
        intentions = ctx.list_intentions(agent_id="BOB")
        intent_ids = {i.id for i in intentions}
        assert intention.id not in intent_ids

        # ALICE should see it
        intentions = ctx.list_intentions(agent_id="ALICE")
        intent_ids = {i.id for i in intentions}
        assert intention.id in intent_ids


# =============================================================================
# TestBackwardsCompat - Single-agent mode works unchanged
# =============================================================================

class TestBackwardsCompat:
    """Verify single-agent and no-filter modes work as before."""

    def test_single_agent_default_shared(self):
        """Default visibility is SHARED, single-agent works unchanged."""
        with tempfile.TemporaryDirectory() as d:
            ctx = CerebroCortex(
                db_path=Path(d) / "compat.db",
                chroma_dir=Path(d) / "chroma",
            )
            ctx.initialize()

            node = ctx.remember(
                content="Single agent memory that should be visible to everyone",
                agent_id="CLAUDE",
            )
            assert node is not None
            assert node.metadata.visibility == Visibility.SHARED

            # Recall without agent_id returns everything
            results = ctx.recall("single agent memory")
            assert len(results) > 0

            # Recall with agent_id also sees shared
            results = ctx.recall("single agent memory", agent_id="CLAUDE")
            assert len(results) > 0

            ctx.close()

    def test_recall_without_agent_id_returns_all(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        results = ctx.recall(
            "Python decorators secret debugging deployment strategy",
            agent_id=None,
        )
        # Should see both shared and private (no filter)
        assert len(results) >= 1


# =============================================================================
# TestShareMemory - Visibility change operations
# =============================================================================

class TestShareMemory:
    """Test the share_memory operation."""

    def test_owner_can_share(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        alice_private = multi_agent_cortex["alice_private"]

        # Alice shares her private memory
        updated = ctx.share_memory(alice_private.id, Visibility.SHARED, agent_id="ALICE")
        assert updated is not None
        assert updated.metadata.visibility == Visibility.SHARED

        # Now BOB can see it
        node = ctx.get_memory(alice_private.id, agent_id="BOB")
        assert node is not None

    def test_non_owner_cannot_share(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        alice_private = multi_agent_cortex["alice_private"]

        # BOB tries to share Alice's memory - should fail
        result = ctx.share_memory(alice_private.id, Visibility.SHARED, agent_id="BOB")
        assert result is None

    def test_share_syncs_chromadb(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        alice_private = multi_agent_cortex["alice_private"]

        # Share it
        updated = ctx.share_memory(alice_private.id, Visibility.SHARED, agent_id="ALICE")
        assert updated is not None

        # Verify ChromaDB was updated (recall should now find it for BOB)
        results = ctx.recall(
            "secret configuration API keys",
            agent_id="BOB",
        )
        result_ids = {node.id for node, _ in results}
        assert alice_private.id in result_ids


# =============================================================================
# TestCrossAgentLinks - Auto-link respects scope
# =============================================================================

class TestCrossAgentLinks:
    """Test that auto-link creation respects scope."""

    def test_auto_link_skips_others_private(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]

        # Store a new memory as BOB with same tags as Alice's private memory
        bob_mem = ctx.remember(
            content="Bob's knowledge about configuration management and API design",
            tags=["config", "secret"],
            agent_id="BOB",
            visibility=Visibility.SHARED,
        )
        assert bob_mem is not None

        # Check that BOB's memory is NOT linked to Alice's private memory
        neighbors = ctx.links.get_neighbors(bob_mem.id)
        neighbor_ids = {n[0] for n in neighbors}
        assert multi_agent_cortex["alice_private"].id not in neighbor_ids


# =============================================================================
# TestScopeSql - Unit tests for _scope_sql helper
# =============================================================================

class TestScopeSql:
    """Test the _scope_sql helper function."""

    def test_no_agent_returns_empty(self):
        clause, params = _scope_sql(None)
        assert clause == ""
        assert params == []

    def test_agent_without_thread(self):
        clause, params = _scope_sql("ALICE")
        assert "visibility='shared'" in clause
        assert "visibility='private'" in clause
        assert "visibility='thread'" in clause
        assert len(params) == 2
        assert params[0] == "ALICE"
        assert params[1] == "ALICE"

    def test_agent_with_thread(self):
        clause, params = _scope_sql("ALICE", "thread-42")
        assert "conversation_thread=?" in clause
        assert len(params) == 3
        assert params[2] == "thread-42"


# =============================================================================
# TestGetDeleteAccess - CRUD scope enforcement
# =============================================================================

class TestGetDeleteAccess:
    """Test get/delete/update with agent scope checks."""

    def test_get_memory_with_access(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        node = ctx.get_memory(multi_agent_cortex["alice_shared"].id, agent_id="BOB")
        assert node is not None

    def test_get_memory_without_access(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        node = ctx.get_memory(multi_agent_cortex["alice_private"].id, agent_id="BOB")
        assert node is None

    def test_get_memory_no_agent_sees_all(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        node = ctx.get_memory(multi_agent_cortex["alice_private"].id)
        assert node is not None

    def test_delete_memory_with_access(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        # Alice can delete her shared memory
        success = ctx.delete_memory(multi_agent_cortex["alice_shared"].id, agent_id="ALICE")
        assert success

    def test_delete_memory_without_access(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        # BOB cannot delete Alice's private memory
        success = ctx.delete_memory(multi_agent_cortex["alice_private"].id, agent_id="BOB")
        assert not success

    def test_update_memory_without_access(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        # BOB cannot update Alice's private memory
        result = ctx.update_memory(
            multi_agent_cortex["alice_private"].id,
            salience=0.1,
            agent_id="BOB",
        )
        assert result is None
