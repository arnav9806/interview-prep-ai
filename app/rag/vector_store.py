# app/rag/vector_store.py

import faiss
import numpy as np
import os

VECTOR_DB_PATH = "vector_db/resume_faiss.index"


class ResumeVectorStore:
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        print(f"FAISS index created with dimension {embedding_dim}")

    def add_embeddings(self, embeddings):
        embeddings_np = np.array(embeddings).astype("float32")
        self.index.add(embeddings_np)
        print(f"Added {len(embeddings)} embeddings to FAISS")

    def save_index(self):
        os.makedirs("vector_db", exist_ok=True)
        faiss.write_index(self.index, VECTOR_DB_PATH)
        print(f"FAISS index saved at {VECTOR_DB_PATH}")

    def load_index(self):
        if os.path.exists(VECTOR_DB_PATH):
            self.index = faiss.read_index(VECTOR_DB_PATH)
            print(f"FAISS index loaded from {VECTOR_DB_PATH}")
        else:
            print("No FAISS index found, creating a new one")

    def search(self, query_embedding, top_k=3):
        query_np = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_np, top_k)
        return distances[0], indices[0]