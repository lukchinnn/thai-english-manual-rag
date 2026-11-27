from .loader import load_pdfs
from .embedder import embed
from .stores import get_store

import time
from .llm import generate

SYSTEM = """คุณคือผู้ช่วยตอบคำถามจากคู่มือเทคนิค
กฎ:
1. ตอบจาก CONTEXT เท่านั้น ห้ามใช้ความรู้อื่น
2. ท้ายข้อเท็จจริงทุกข้อให้อ้างอิงแบบ [ชื่อไฟล์ p.เลขหน้า]
3. ถ้า CONTEXT ไม่มีคำตอบ ให้ตอบว่า "ไม่พบข้อมูลในเอกสาร"
4. ตอบภาษาเดียวกับคำถาม"""

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

def ask(question: str, k: int = 5) -> dict:
    t0 = time.perf_counter()
    hits = retrieve(question, k)
    t1 = time.perf_counter()
    context = "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits)
    answer = generate(SYSTEM, f"CONTEXT:\n{context}\n\nQUESTION: {question}")
    t2 = time.perf_counter()
    return {
        "answer": answer,
        "sources": [{"source": h["source"], "page": h["page"], "score": h["score"]} for h in hits],
        "contexts": [h["text"] for h in hits],
        "latency": {"retrieve_s": round(t1 - t0, 3), "generate_s": round(t2 - t1, 3)},
    }