"""Re-embed CerebroCortex ChromaDB collections with the currently-configured embedder.

Use when switching embedder models (e.g., sentence-transformers MiniLM -> BGE).
Idempotent: re-running just re-embeds everything again, still safe.
Also stamps each collection with the current embedder's fingerprint so the
on-boot fingerprint check treats the collection as up-to-date afterward.
"""
import sys
import chromadb

from cerebro.settings import load_on_startup
from cerebro.storage.embedder_fingerprint import fingerprint_for, merge_into_metadata, strip_immutable_for_modify
from cerebro.storage.embeddings import get_embedding_function

CHROMA_PATH = "/home/andre/.cerebro-cortex/chroma"


def main() -> int:
    load_on_startup()
    ef = get_embedding_function("auto")
    fp = fingerprint_for(ef)
    print(f"[reembed] using: {ef.name()}  dim={fp['cc_embedder_dim']}")
    print(f"[reembed] fingerprint: {fp}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    colls = client.list_collections()
    print(f"[reembed] found {len(colls)} collections")

    for coll in colls:
        n = coll.count()
        if n == 0:
            # empty — just stamp the fingerprint so future inserts match
            new_meta = merge_into_metadata(strip_immutable_for_modify(coll.metadata or {}), fp)
            coll.modify(metadata=new_meta)
            print(f"  {coll.name}: empty, fingerprint stamped")
            continue
        print(f"  {coll.name}: {n} items -> re-embedding")
        data = coll.get(include=["documents", "metadatas"])
        ids = data["ids"]
        docs = data["documents"]
        # ChromaDB: .update() accepts new embeddings for existing ids
        new_vecs = ef.embed(docs)
        coll.update(ids=ids, embeddings=[v.tolist() for v in new_vecs])
        # Stamp the fingerprint onto the collection metadata too
        new_meta = merge_into_metadata(strip_immutable_for_modify(coll.metadata or {}), fp)
        coll.modify(metadata=new_meta)
        print(f"    [ok] re-embedded {len(ids)} items + fingerprint stamped")

    print("[reembed] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
