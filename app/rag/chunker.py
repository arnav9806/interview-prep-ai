# app/rag/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text):

    print("Starting text chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    print(f"Total chunks created: {len(chunks)}")

    return chunks