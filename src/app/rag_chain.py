import os

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:14b")

class RAGPipeline:
    def __init__(self):
        self.llm = OllamaLLM(
            base_url=OLLAMA_URL, 
            model=MODEL_NAME,
            temperature=0.0 
        )
        self.embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
        
        self.client = QdrantClient(host=QDRANT_HOST, port=6333)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name="knowledge_base",
            embedding=self.embeddings
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an exact factual AI assistant.\n"
                "Answer the user question using ONLY the provided context snippets below.\n"
                "DO NOT introduce outside assumptions, external facts, or summary opinions not explicitly stated in the context.\n"
                "If the context does not explicitly answer the question, state: 'I cannot answer based on the provided context.'\n\n"
                "Context:\n{context}"
            )),
            ("human", "{input}"),
        ])
        
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    def query(self, user_prompt: str):
        response = self.rag_chain.invoke({"input": user_prompt})
        return {
            "answer": response["answer"],
            "context": [doc.page_content for doc in response["context"]]
        }