from pathlib import Path
import uuid
import json as json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from supabase import create_client
from pinecone_text.sparse import BM25Encoder
from pathlib import Path

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
bm25 = BM25Encoder()

# in-memory parent store for current session
parent_store = {}


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def chunk_document(docs, document_id):
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    parent_chunks = parent_splitter.split_documents(docs)
    all_child_chunks = []

    for p_idx, parent_chunk in enumerate(parent_chunks):
        parent_id = f"{document_id}_p{p_idx}"
        parent_store[parent_id] = parent_chunk.page_content  # fill in-memory store

        children = child_splitter.split_documents([parent_chunk])
        for c_idx, child in enumerate(children):
            child.metadata["parent_id"] = parent_id
            child.metadata["chunk_id"] = f"{parent_id}_c{c_idx}"
            child.metadata["document_id"] = document_id
            all_child_chunks.append(child)

    return all_child_chunks


# ── Parent Store Supabase ─────────────────────────────────────────────────────

def save_parent_store_to_supabase(user_id: str, session_id: str, source: str):
    """Save current in-memory parent_store to Supabase."""
    rows = [
        {
            "id": pid,
            "content": text,
            "user_id": user_id,
            "session_id": session_id,
            "source": source
        }
        for pid, text in parent_store.items()
    ]
    if rows:
        supabase.table("parent_store").upsert(rows).execute()
        print(f"Parent store saved: {len(rows)} parents")


def load_parent_store_from_supabase(user_id: str, session_id: str):
    """Load parent store from Supabase into memory."""
    global parent_store
    result = supabase.table("parent_store") \
        .select("id, content") \
        .eq("user_id", user_id) \
        .eq("session_id", session_id) \
        .execute()

    if result.data:
        parent_store = {row["id"]: row["content"] for row in result.data}
        print(f"Parent store loaded: {len(parent_store)} parents")
    else:
        print("No parent store found in Supabase.")


# ── BM25 ──────────────────────────────────────────────────────────────────────

def train_and_save_bm25(texts: list[str], user_id: str = "default"):
    global bm25
    bm25.fit(texts)
    params = bm25.get_params()
    supabase.table("bm25_store").upsert({
        "id": user_id,
        "params": json.dumps(params)
    }).execute()
    print(f"BM25 saved for user: {user_id}")


def load_bm25_from_supabase(user_id="default"):
    global bm25
    result = supabase.table("bm25_store") \
        .select("params") \
        .eq("id", user_id) \
        .execute()

    if result.data:
        params = json.loads(result.data[0]["params"])
        bm25 = BM25Encoder()
        bm25.set_params(**params)
        print(f"BM25 loaded for user: {user_id}")
        return True

    print("No BM25 params found.")
    return False

def get_session_texts_from_pinecone(session_id: str) -> list:
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    all_texts = []
    
    results = index.query(
        vector=[0.0] * 1024,
        top_k=10000,
        include_metadata=True,
        filter={"session_id": {"$eq": session_id}}
    )
    
    for match in results["matches"]:
        text = match["metadata"].get("text", "")
        if text:
            all_texts.append(text)
    
    return all_texts

# ── Embeddings ────────────────────────────────────────────────────────────────

def embedding_text(texts, input_type="passage"):
    embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=texts,
        parameters={
            "input_type": input_type,
            "truncate": "END"
        }
    )
    return embeddings


# ── Pinecone ──────────────────────────────────────────────────────────────────

def create_index_if_not_exists():
    index_name = os.getenv("PINECONE_INDEX_NAME")
    existing = [i.name for i in pc.indexes.list()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)


def pinecone_store(chunks, embeddings, session_id: str = "default"):
    index = create_index_if_not_exists()
    texts = [chunk.page_content for chunk in chunks]
    sparse_embeddings = bm25.encode_documents(texts)
    vectors = []
    for chunk, dense_emb, sparse_emb in zip(chunks, embeddings, sparse_embeddings):
        vectors.append({
            "id": chunk.metadata["chunk_id"],
            "values": dense_emb["values"],
            "sparse_values": sparse_emb,
            "metadata": {
                "session_id": session_id,
                "document_id": chunk.metadata["document_id"],
                "chunk_id": chunk.metadata["chunk_id"],
                "parent_id": chunk.metadata["parent_id"],
                "source": chunk.metadata["source"],
                "page": chunk.metadata.get("page", 0),
                "text": chunk.page_content
            }
        })
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100])


def is_document_already_indexed(filename: str, session_id: str = "default") -> bool:
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    results = index.query(
        vector=[0.0] * 1024,
        top_k=1,
        include_metadata=True,
        filter={
            "source": {"$eq": filename},
            "session_id": {"$eq": session_id}
        }
    )
    return len(results["matches"]) > 0


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrive(query, top_k=5, alpha=0.5, source_filter=None, session_id: str = "default"):
    dense_vector = embedding_text([query], input_type='query')[0]["values"]
    sparse_vector = bm25.encode_queries(query)

    scaled_dense = [v * alpha for v in dense_vector]
    scaled_sparse = {
        "indices": sparse_vector["indices"],
        "values": [v * (1 - alpha) for v in sparse_vector["values"]]
    }

    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    pinecone_filter = {"session_id": {"$eq": session_id}}
    if source_filter:
        pinecone_filter["source"] = {"$eq": source_filter}

    query_kwargs = {
        "vector": scaled_dense,
        "sparse_vector": scaled_sparse,
        "top_k": 20,
        "include_metadata": True,
        "filter": pinecone_filter
    }

    results = index.query(**query_kwargs)
    matches = results["matches"]

    if matches:
        documents = [m["metadata"]["text"] for m in matches]
        reranked = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=True
        )
        reranked_matches = [matches[r["index"]] for r in reranked.data]
    else:
        reranked_matches = matches

    contexts = []
    seen_parents = set()
    for match in reranked_matches:
        parent_id = match["metadata"].get("parent_id")
        if parent_id and parent_id not in seen_parents:
            # try in-memory first, then Pinecone child text as fallback
            parent_text = parent_store.get(parent_id, match["metadata"].get("text", ""))
            contexts.append(parent_text)
            seen_parents.add(parent_id)

    return contexts

def ingest_pdf(pdf_path: str, user_id: str, session_id: str):
    
    filename = Path(pdf_path).name
    bm25_id = f"{user_id}_{session_id}"
    
    # Skip if already indexed
    if is_document_already_indexed(filename, session_id=session_id):
        print(f"'{filename}' already indexed, skipping.")
        return
    
    docs = load_pdf(pdf_path)
    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    
    for doc in docs:
        doc.metadata["document_id"] = document_id
        doc.metadata["source"] = filename
    
    child_chunks = chunk_document(docs, document_id)
    existing_texts = get_session_texts_from_pinecone(session_id)
    new_texts = [chunk.page_content for chunk in child_chunks]
    all_texts = existing_texts + new_texts
    
    save_parent_store_to_supabase(user_id, session_id, filename)
    train_and_save_bm25(all_texts, user_id=bm25_id)
    
    texts = [chunk.page_content for chunk in child_chunks]
    dense_embedding = embedding_text(texts)
    pinecone_store(child_chunks, dense_embedding, session_id=session_id)
    
    print(f"Ingestion complete for {filename}")

def hybrid_rag_node(state):
    user_query = state["messages"][-1].content
    user_id = state.get("user_id", "default")
    session_id = state.get("session_id", "default")
    bm25_id = f"{user_id}_{session_id}"

    # Load from Supabase — no file loading
    load_parent_store_from_supabase(user_id, session_id)
    load_bm25_from_supabase(user_id=bm25_id)
    create_index_if_not_exists()

    context = retrive(
        query=user_query,
        top_k=5,
        alpha=0.5,
        source_filter=None,
        session_id=session_id
    )

    return {
        "retrieved_context": context,
        "current_context": context,
    }