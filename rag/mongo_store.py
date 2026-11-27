from pymongo import MongoClient, UpdateOne
from pymongo.operations import SearchIndexModel
from .config import MONGODB_URI

class MongoStore:
    def __init__(self, db: str = "rag", col: str = "chunks", index: str = "vector_index"):
        self.col = MongoClient(MONGODB_URI)[db][col]
        self.index = index

    def add(self, chunks: list[dict], vectors: list[list[float]]):
        ops = [UpdateOne({"_id": c["id"]},
                         {"$set": {"text": c["text"], "source": c["source"],
                                   "page": c["page"], "embedding": v}},
                         upsert=True)
               for c, v in zip(chunks, vectors)]
        self.col.bulk_write(ops)

    def search(self, qvec: list[float], k: int = 5, source: str | None = None) -> list[dict]:
        stage = {"index": self.index, "path": "embedding", "queryVector": qvec,
                 "numCandidates": k * 20, "limit": k}
        if source:
            stage["filter"] = {"source": source}
        pipeline = [
            {"$vectorSearch": stage},
            {"$project": {"_id": 0, "text": 1, "source": 1, "page": 1,
                          "score": {"$meta": "vectorSearchScore"}}},
        ]
        return list(self.col.aggregate(pipeline))

    def create_index(self, dims: int = 1024):
        self.col.create_search_index(SearchIndexModel(
            name=self.index, type="vectorSearch",
            definition={"fields": [
                {"type": "vector", "path": "embedding", "numDimensions": dims, "similarity": "cosine"},
                {"type": "filter", "path": "source"}]}))