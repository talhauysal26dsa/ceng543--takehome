from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .data import Document

@dataclass
class RetrievedDocument:
    doc_id: str
    score: float

class BM25Retriever:
    def __init__(self, documents: Sequence[Document]):
        self.doc_ids = [doc.doc_id for doc in documents]
        # Include title in tokenization for better matching
        tokenized = [f"{doc.title} {doc.text}".lower().split() for doc in documents]
        self.model = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        # Lowercase query for consistency
        scores = self.model.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievedDocument(doc_id=self.doc_ids[i], score=float(scores[i]))
            for i in top_indices
        ]

class DenseRetriever:
        def __init__(self, documents: Sequence[Document], model_name: str = "all-MiniLM-L6-v2"):
        self.doc_ids = [doc.doc_id for doc in documents]
        self.model = SentenceTransformer(model_name)
        
        # Encode all documents (title + text for better retrieval)
        doc_texts = [f"{doc.title}. {doc.text}" for doc in documents]
        print(f"Encoding {len(doc_texts)} documents with {model_name}...")
        self.doc_embeddings = self.model.encode(doc_texts, show_progress_bar=True, convert_to_numpy=True)
        self.doc_embeddings = self.doc_embeddings / np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
    
    def search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Compute cosine similarity
        scores = np.dot(self.doc_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [
            RetrievedDocument(doc_id=self.doc_ids[i], score=float(scores[i]))
            for i in top_indices
        ]

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "RetrievedDocument",
]
