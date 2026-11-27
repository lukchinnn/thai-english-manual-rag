import chromadb

class ChromaStore:
    def __init__(self, path: str = "./chroma_db", name: str = "manuals"):
        self.col = chromadb.PersistentClient(path=path).get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}  # เวกเตอร์ normalize แล้ว ใช้ cosine ให้ score = cos จริง
        )

    def add(self, chunks: list[dict], vectors: list[list[float]]):
        self.col.upsert(  # upsert = รันซ้ำได้ไม่เกิดข้อมูลซ้ำ
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
            embeddings=vectors,
        )

    def search(self, qvec: list[float], k: int = 5) -> list[dict]:
        r = self.col.query(query_embeddings=[qvec], n_results=k)
        return [
            {"text": doc, "source": meta["source"], "page": meta["page"], "score": 1 - dist}
            for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]

def get_store():
    from .config import VECTOR_BACKEND
    if VECTOR_BACKEND == "mongo":
        from .mongo_store import MongoStore
        return MongoStore()
    return ChromaStore()