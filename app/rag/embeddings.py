# app/rag/embeddings.py

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

# ✅ NEW WAY (NO configure)
client = genai.Client(api_key=API_KEY)

print("✅ Gemini configured successfully")

def normalize_embeddings(embeddings):
    return np.array([
        e / np.linalg.norm(e) if np.linalg.norm(e) != 0 else e
        for e in embeddings
    ])

# ======================================
# 🚀 CREATE EMBEDDINGS
# ======================================
def create_embeddings(chunks):
    texts = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"] if isinstance(chunk, dict) else chunk
        texts.append(text)

        print(f"\n🧩 Chunk {i} | Length: {len(text)}")
        print(f"Preview: {text[:100]}")

    print(f"\n📊 Total chunks to embed: {len(texts)}")

    try:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )

        embeddings = [e.values for e in response.embeddings]

        print("✅ Batch embedding successful")

    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return []

    # 🔥 NORMALIZATION
    embeddings = normalize_embeddings(embeddings)

    # 🔥 ATTACH METADATA
    for i in range(len(chunks)):
        chunks[i]["embedding"] = embeddings[i]

    print("\n✅ EMBEDDING COMPLETED")
    print(f"Total embeddings: {len(embeddings)}")

    return chunks