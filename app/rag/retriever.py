# app/rag/retriever.py

from app.rag.vector_store import ResumeVectorStore
from app.rag.embeddings import create_embeddings


class ResumeRetriever:

    def __init__(self):
        print("Initializing ResumeRetriever...")

        # Load FAISS index
        self.vector_store = ResumeVectorStore()
        self.vector_store.load_index()

        print("FAISS index loaded successfully")

    def retrieve_chunks(self, query_text, top_k=3):
        """
        Retrieve most relevant resume chunks using FAISS
        """
        print("Creating embedding for query...")
        # Create embedding for query
        query_embedding = create_embeddings([query_text])[0]
        print("🔎 Searching FAISS index...")

        distances, indices = self.vector_store.search(query_embedding, top_k)
        print("Retrieved chunk indices:", indices)
        print("Distances:", distances)

        return indices, distances
