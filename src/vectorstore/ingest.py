import os

from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore as Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")

def seed_vector_db():
    print("Processing knowledge base documents...")
    
    loader = TextLoader("sample_knowledge.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs = text_splitter.split_documents(documents)

    print("Generating embeddings via Ollama...")
    embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")

    print("Storing vectors in Qdrant...")
    Qdrant.from_documents(
        docs,
        embeddings,
        host=QDRANT_HOST,
        port=6333,
        collection_name="knowledge_base",
        force_recreate=True
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    seed_vector_db()