# app/rag/vector_store.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# ======================================
# 🔐 QDRANT CONFIG
# ======================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "resume_chunks"

# ======================================
# 🚀 INIT CLIENT
# ======================================
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("\n🔗 Qdrant Client Initialized")


# ======================================
# 🧪 DEBUG: CONNECTION CHECK
# ======================================
def debug_qdrant():
    collections = client.get_collections()
    print("\n📦 QDRANT CONNECTION ACTIVE")
    print("Available Collections:")
    for c in collections.collections:
        print(" -", c.name)


# ======================================
# 📦 CREATE COLLECTION
# ======================================
def create_collection(vector_size=768):
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"\n✅ Collection '{COLLECTION_NAME}' created in Qdrant")
    else:
        print(f"\n✅ Collection '{COLLECTION_NAME}' already exists")


# ======================================
# 📥 ADD CHUNKS (UPSERT)
# ======================================
def add_chunks(chunks, embeddings):
    points = []

    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):

        text = chunk["text"] if isinstance(chunk, dict) else chunk
        section = chunk.get("section", "general") if isinstance(chunk, dict) else "general"

        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "text": text,
                    "section": section
                }
            )
        )

    result = client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("\n📤 UPLOAD COMPLETED")
    print("Upsert response:", result)
    print("Total chunks uploaded:", len(points))


# ======================================
# 🔎 SEARCH FUNCTION
# ======================================
def search(query_vector, top_k=3):
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    output = []

    print("\n🔍 QDRANT SEARCH RESULTS:")

    for i, r in enumerate(results):
        print(f"\nResult {i}")
        print("Score:", r.score)
        print("Section:", r.payload.get("section"))
        print("Text:", r.payload.get("text")[:150])

        output.append({
            "score": r.score,
            "text": r.payload.get("text"),
            "section": r.payload.get("section")
        })

    return output


# ======================================
# 🔎 DEBUG: CHECK STORED DATA
# ======================================
def check_collection(limit=5):
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )

    print("\n📊 STORED VECTOR DATA CHECK:")

    if not points:
        print("❌ No data found in collection!")
        return

    for i, p in enumerate(points):
        print(f"\nPoint {i}")
        print("ID:", p.id)
        print("Section:", p.payload.get("section"))
        print("Preview:", p.payload.get("text")[:120])


# ======================================
# 🧪 FULL SYSTEM DEBUG RUN
# ======================================
def run_debug_pipeline(chunks, embeddings):
    print("\n🚀 RUNNING QDRANT DEBUG PIPELINE")

    debug_qdrant()
    create_collection(vector_size=len(embeddings[0]))
    add_chunks(chunks, embeddings)
    check_collection()



# import os
# from dotenv import load_dotenv
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct

# load_dotenv()

# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# # ======================================
# # 🚀 INIT QDRANT CLIENT (CLOUD)
# # ======================================
# client = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY
# )

# COLLECTION_NAME = "resume_chunks"

# def debug_qdrant():
#     collections = client.get_collections()
#     print("\n📦 QDRANT CONNECTION ACTIVE")
#     print("Collections:", [c.name for c in collections.collections])
    
# debug_qdrant()

# # ======================================
# # 📦 CREATE COLLECTION
# # ======================================
# def create_collection(vector_size=768):
#     existing = [c.name for c in client.get_collections().collections]

#     if COLLECTION_NAME not in existing:
#         client.create_collection(
#             collection_name=COLLECTION_NAME,
#             vectors_config=VectorParams(
#                 size=vector_size,
#                 distance=Distance.COSINE
#             )
#         )
#         print("✅ Qdrant collection created")
#     else:
#         print("✅ Collection already exists")


# # ======================================
# # 📥 ADD CHUNKS
# # ======================================
# def add_chunks(chunks, embeddings):
#     points = []

#     for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):

#         text = chunk["text"] if isinstance(chunk, dict) else chunk

#         points.append(
#             PointStruct(
#                 id=i,
#                 vector=vector,
#                 payload={
#                     "text": text,
#                     "section": chunk.get("section", "general") if isinstance(chunk, dict) else "general"
#                 }
#             )
#         )

#     client.upsert(
#         collection_name=COLLECTION_NAME,
#         points=points
#     )

#     print(f"✅ Added {len(points)} chunks to Qdrant")


# # ======================================
# # 🔎 SEARCH
# # ======================================
# def search(query_vector, top_k=3):
#     results = client.search(
#         collection_name=COLLECTION_NAME,
#         query_vector=query_vector,
#         limit=top_k
#     )

#     output = []
#     for r in results:
#         output.append({
#             "score": r.score,
#             "text": r.payload["text"],
#             "section": r.payload["section"]
#         })
#     print("Output>>>>>>>>>>>>", output)

#     return output