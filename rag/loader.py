from pathlib import Path
from pypdf import PdfReader

def split_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())          # ยุบช่องว่าง/ขึ้นบรรทัดซ้ำ
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text), 1), step) if text[i:i + size].strip()]

def load_pdfs(folder: str = "data", size: int = 800, overlap: int = 150) -> list[dict]:
    chunks = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
            for j, piece in enumerate(split_text(page.extract_text() or "", size, overlap)):
                chunks.append({
                    "id": f"{pdf.stem}-p{page_no}-{j}",
                    "text": piece,
                    "source": pdf.name,
                    "page": page_no,
                })
    return chunks