from rag.loader import load_pdfs
from rag.embedder import embed
import numpy as np

chunks = load_pdfs()
print(len(chunks), chunks[0])            # ดูว่าตัดออกมาอ่านรู้เรื่องไหม
a, b, c = embed(["มอเตอร์ร้อนเกิน", "motor overheating", "กาแฟเย็น"])
print(np.dot(a, b), np.dot(a, c))         # คู่แรกควรสูงกว่าชัดเจน