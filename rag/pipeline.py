from .loader import load_pdfs
from .embedder import embed
from .stores import get_store

def ingest(folder: str = "data"):
    chunks = load_pdfs(folder)
    vectors = embed([c["text"] for c in chunks])
    get_store().add(chunks, vectors)
    print(f"indexed {len(chunks)} chunks")

def retrieve(question: str, k: int = 5) -> list[dict]:
    return get_store().search(embed([question])[0], k)

if __name__ == "__main__":
    ingest()
    for hit in retrieve("วิธีแก้เมื่ออุปกรณ์ร้อนเกิน"):
        print(round(hit["score"], 3), hit["source"], hit["page"], hit["text"][:80])