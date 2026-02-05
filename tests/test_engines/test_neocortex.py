"""Tests for the SchemaEngine (neocortex)."""

from cerebro.engines.neocortex import SchemaEngine
from cerebro.models.memory import MemoryMetadata, MemoryNode
from cerebro.types import MemoryLayer, MemoryType


class TestSchemaCreation:
    def test_create_schema(self, graph_store):
        engine = SchemaEngine(graph_store)

        # Add source memories
        for i in range(3):
            graph_store.add_node(MemoryNode(
                id=f"mem_schema_src_{i}",
                content=f"Episode about deployment pattern {i}",
            ))

        schema = engine.create_schema(
            content="Deployments follow: build -> test -> stage -> deploy",
            source_ids=[f"mem_schema_src_{i}" for i in range(3)],
            tags=["deployment", "pattern"],
        )

        assert schema.metadata.memory_type == MemoryType.SCHEMATIC
        assert schema.metadata.layer == MemoryLayer.LONG_TERM
        assert schema.metadata.salience == 0.9
        assert schema.strength.stability == 30.0

    def test_derived_from_links(self, graph_store):
        engine = SchemaEngine(graph_store)

        graph_store.add_node(MemoryNode(id="mem_df1", content="Source 1"))
        graph_store.add_node(MemoryNode(id="mem_df2", content="Source 2"))

        schema = engine.create_schema(
            content="Pattern from sources",
            source_ids=["mem_df1", "mem_df2"],
        )

        # Should have derived_from links to sources
        assert graph_store.has_link(schema.id, "mem_df1")
        assert graph_store.has_link(schema.id, "mem_df2")


class TestSchemaRetrieval:
    def test_find_by_tags(self, graph_store):
        engine = SchemaEngine(graph_store)
        engine.create_schema(
            content="Debugging pattern",
            source_ids=[],
            tags=["debugging"],
        )
        engine.create_schema(
            content="Deployment pattern",
            source_ids=[],
            tags=["deployment"],
        )

        results = engine.find_matching_schemas(tags=["debugging"])
        assert len(results) >= 1
        assert any("debugging" in s.metadata.tags for s in results)

    def test_find_by_concepts(self, graph_store):
        engine = SchemaEngine(graph_store)
        # Create schema with concepts
        node = MemoryNode(
            id="mem_schema_c1",
            content="API design follows REST principles",
            metadata=MemoryMetadata(
                memory_type=MemoryType.SCHEMATIC,
                layer=MemoryLayer.LONG_TERM,
                concepts=["api", "rest", "design"],
                salience=0.9,
            ),
        )
        graph_store.add_node(node)

        results = engine.find_matching_schemas(concepts=["api"])
        assert len(results) >= 1

    def test_get_schema_sources(self, graph_store):
        engine = SchemaEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_src_a", content="Source A"))

        schema = engine.create_schema(
            content="Derived schema",
            source_ids=["mem_src_a"],
        )

        sources = engine.get_schema_sources(schema.id)
        assert "mem_src_a" in sources


class TestSchemaReinforcement:
    def test_reinforce_schema(self, graph_store):
        engine = SchemaEngine(graph_store)
        schema = engine.create_schema(
            content="Users prefer dark mode",
            source_ids=[],
        )
        original_salience = schema.metadata.salience

        # Add supporting evidence
        graph_store.add_node(MemoryNode(id="mem_evidence1", content="User asked for dark mode"))

        result = engine.reinforce_schema(schema.id, "mem_evidence1")
        assert result is True

        # Salience should increase
        updated = graph_store.get_node(schema.id)
        assert updated.metadata.salience >= original_salience

        # Should have supports link
        assert graph_store.has_link("mem_evidence1", schema.id)

    def test_reinforce_nonexistent(self, graph_store):
        engine = SchemaEngine(graph_store)
        result = engine.reinforce_schema("nonexistent", "also_nonexistent")
        assert result is False


class TestSchemaStats:
    def test_count_schemas(self, graph_store):
        engine = SchemaEngine(graph_store)
        assert engine.count_schemas() == 0

        engine.create_schema("Schema 1", source_ids=[])
        engine.create_schema("Schema 2", source_ids=[])

        assert engine.count_schemas() == 2

    def test_get_all_schemas(self, graph_store):
        engine = SchemaEngine(graph_store)
        engine.create_schema("Schema A", source_ids=[], tags=["a"])
        engine.create_schema("Schema B", source_ids=[], tags=["b"])

        all_schemas = engine.get_all_schemas()
        assert len(all_schemas) == 2
