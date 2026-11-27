from functools import lru_cache
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL

@lru_cache
def _model():
    return SentenceTransformer(EMBED_MODEL)

def embed(texts: list[str]) -> list[list[float]]:
    # normalize ให้ความยาว = 1 จะใช้ cosine หรือ dot product ก็ได้ผลเรียงเหมือนกัน
    return _model().encode(texts, normalize_embeddings=True,
                           batch_size=16, show_progress_bar=len(texts) > 50).tolist()