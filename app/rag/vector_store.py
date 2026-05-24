# app/rag/vector_store.py

import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue

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

        print(f"\n✅ Collection '{COLLECTION_NAME}' created")

        # 🔥 ADD THESE LINES
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source",
            field_schema="keyword"
        )

        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="section",
            field_schema="keyword"
        )

        print("✅ Payload indexes created")

    else:
        print(f"\n✅ Collection already exists")
# ======================================
# 📥 ADD CHUNKS (UPSERT)
# ======================================
def add_chunks(chunks, embeddings, source="resume"):
    points = []

    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):

        text = chunk["text"]
        metadata = chunk.get("metadata", {})

        section = metadata.get("section", "general")
        chunk_id = metadata.get("chunk_id", i)

        # ✅ FIX: Use UUID or integer (recommended UUID)
        point_id = str(uuid.uuid4())

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": text,
                    "section": section,
                    "chunk_id": chunk_id,
                    "source": source   # 🔥 IMPORTANT
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
def search(query_vector, top_k=3, source=None, section=None):

    filters = []

    if source:
        filters.append(
            FieldCondition(
                key="source",
                match=MatchValue(value=source)
            )
        )

    if section:
        filters.append(
            FieldCondition(
                key="section",
                match=MatchValue(value=section)
            )
        )

    query_filter = None
    if filters:
        query_filter = Filter(must=filters)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k
    ).points

    output = []

    print("\n🔍 QDRANT SEARCH RESULTS:")

    for i, r in enumerate(results):

        base_score = r.score
        section = r.payload.get("section")
        source = r.payload.get("source")

        # 🔥 BOOSTING
        boost = 1.0

        if source == "resume":
            boost += 0.2

        if section in ["skills", "experience"]:
            boost += 0.3

        final_score = base_score * boost

        print(f"\nResult {i}")
        print("Base Score:", base_score)
        print("Final Score:", final_score)
        print("Section:", section)
        print("Source:", source)

        output.append({
            "score": final_score,
            "text": r.payload.get("text"),
            "section": section,
            "source": source
        })

    # 🔥 SORT AFTER BOOST
    output = sorted(output, key=lambda x: x["score"], reverse=True)

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
        print("Section:", p.payload.get("section"))
        print("Source:", p.payload.get("source"))   


# ======================================
# 🧪 FULL SYSTEM DEBUG RUN
# ======================================
def run_debug_pipeline(chunks, embeddings):
    print("\n🚀 RUNNING QDRANT DEBUG PIPELINE")

    debug_qdrant()
    create_collection(vector_size=len(embeddings[0]))
    add_chunks(chunks, embeddings)
    check_collection()

