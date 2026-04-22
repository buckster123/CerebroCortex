"""Re-embed CerebroCortex ChromaDB collections with the currently-configured embedder.

Use when switching embedder models (e.g., sentence-transformers MiniLM -> BGE).
Idempotent: re-running just re-embeds everything again, still safe.
"""
import sys
import chromadb

from cerebro.settings import load_on_startup
from cerebro.storage.embeddings import get_embedding_function
from cerebro import config

CHROMA_PATH = "/home/andre/.cerebro-cortex/chroma"


def main() -> int:
    load_on_startup()
    ef = get_embedding_function("auto")
    print(f"[reembed] using: {ef.name()}  dim={ef.dimension}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    colls = client.list_collections()
    print(f"[reembed] found {len(colls)} collections")

    for coll in colls:
        n = coll.count()
        if n == 0:
            print(f"  {coll.name}: empty, skip")
            continue
        print(f"  {coll.name}: {n} items -> re-embedding")
        data = coll.get(include=["documents", "metadatas"])
        ids = data["ids"]
        docs = data["documents"]
        metas = data["metadatas"]
        # ChromaDB: .update() accepts new embeddings for existing ids
        new_vecs = ef.embed(docs)
        coll.update(ids=ids, embeddings=[v.tolist() for v in new_vecs])
        print(f"    [ok] re-embedded {len(ids)} items")

    print("[reembed] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
