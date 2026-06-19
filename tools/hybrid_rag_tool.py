from pathlib import Path
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def chunk_document(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

def embedding_text(texts):
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(
        "intfloat/multilingual-e5-base"
    )
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    ).tolist()

    return embeddings

def pinecone_store(chunks,embeddings):    
    from pinecone import Pinecone
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
    index = pc.Index(
        os.getenv("PINECONE_INDEX_NAME")
    )
    vectors = []
    for chunk,embedding in zip(chunks,embeddings):
        vectors.append(
            {
                "id": chunk.metadata["chunk_id"],
                "values": embedding,
                "metadata": {
                    "document_id": chunk.metadata["document_id"],
                    "chunk_id": chunk.metadata["chunk_id"],
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata.get("page", 0),
                    "text": chunk.page_content
                }
            }
        )
    index.upsert(vectors=vectors)


def hybrid_rag_node(state):
    pdf_paths = state["pdf_paths"]

    all_chunks = []

    for pdf_path in pdf_paths:
        docs = load_pdf(pdf_path)
        
        document_id = f"doc_{uuid.uuid4().hex[:12]}"

        filename = Path(pdf_path).name

        # Add document metadata
        for doc in docs:
            doc.metadata["document_id"] = document_id
            doc.metadata["source"] = filename
        
        
        chunks = chunk_document(docs)

        for idx,chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = (
                f"{document_id}_{idx}"
            )
        all_chunks.extend(chunks)
    texts = [chunk.page_content for chunk in all_chunks]
    embedding = embedding_text(texts)
    pinecone_store(all_chunks,embedding)


    print("vector store in pinecone !")
    return {}