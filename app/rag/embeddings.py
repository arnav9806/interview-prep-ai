import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ======================================
# 🔐 LOAD ENV
# ======================================
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

print("🔐 Loading Gemini API key...")

client = genai.Client(api_key=API_KEY)

print("✅ Gemini configured successfully")


# ======================================
# 🔧 NORMALIZATION
# ======================================
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector


def normalize_embeddings(embeddings):
    return [normalize_vector(e) for e in embeddings]

# ======================================
# 🚀 CREATE EMBEDDINGS (DOCUMENTS)
# ======================================
def create_embeddings(chunks):
    enriched_chunks = []
    embeddings = []

    print(f"\n📊 Total chunks to embed: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        text = chunk["text"] if isinstance(chunk, dict) else chunk

        if not text or len(text.strip()) == 0:
            print(f"⚠️ Skipping empty chunk {i}")
            continue

        print(f"\n🔹 Embedding chunk {i} | Length: {len(text)}")

        try:
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=[text],   # ✅ SINGLE INPUT ONLY
                config=types.EmbedContentConfig(
                    output_dimensionality=768
                )
            )

            embedding = response.embeddings[0].values

            # normalize
            norm = np.linalg.norm(embedding)
            embedding = embedding / norm if norm != 0 else embedding

        except Exception as e:
            print(f"❌ Failed chunk {i}: {e}")
            continue

        # attach
        if isinstance(chunk, dict):
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = embedding
        else:
            chunk_copy = {
                "text": chunk,
                "embedding": embedding,
                "section": "general"
            }

        enriched_chunks.append(chunk_copy)
        embeddings.append(embedding)

    print("\n✅ EMBEDDING COMPLETED")
    print(f"Total embeddings: {len(embeddings)}")

    if embeddings:
        print(f"Embedding dimension: {len(embeddings[0])}")
    else:
        print("❌ No embeddings created")

    return enriched_chunks, embeddings


# ======================================
# 🔍 CREATE QUERY EMBEDDING
# ======================================
def create_query_embedding(text):
    print(f"\n🔎 Creating query embedding...")
    print(f"Query: {text}")

    try:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=[text],
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )

        embedding = response.embeddings[0].values

        # 🔥 NORMALIZE
        embedding = normalize_vector(embedding)

        print("✅ Query embedding created")
        print(f"Dimension: {len(embedding)}")

        return embedding

    except Exception as e:
        print(f"❌ Query embedding failed: {e}")
        return None
