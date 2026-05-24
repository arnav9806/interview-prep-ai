from app.rag.embeddings import create_query_embedding
from app.rag.vector_store import search


class ResumeRetriever:

    def __init__(self):
        print("🚀 Initializing Qdrant Retriever...")

    def retrieve_chunks(
        self,
        query_text,
        top_k=5,
        source=None,        # "resume" or "jd"
        section=None        # "skills", "experience", etc
    ):
        """
        Retrieve relevant chunks from Qdrant
        """

        print("\n🔎 STEP 1: Creating query embedding...")
        query_vector = create_query_embedding(query_text)

        if query_vector is None:
            print("❌ Query embedding failed")
            return []

        print("✅ Query embedding ready")

        print("\n🔎 STEP 2: Searching Qdrant...")

        results = search(
            query_vector=query_vector,
            top_k=top_k,
            source=source,
            section=section
                
        )

        print(f"✅ Retrieved {len(results)} chunks")

        return results