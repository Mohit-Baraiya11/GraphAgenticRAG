import networkx as nx
import json as _json
import os
from pinecone import Pinecone
from supabase import create_client
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

graph = nx.Graph()


# ── Schemas ───────────────────────────────────────────────────────────────────

class Entity(BaseModel):
    name: str
    type: str

class Relationship(BaseModel):
    source: str
    target: str
    relation: str

class ExtractionResult(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


# ── Entity Extraction ─────────────────────────────────────────────────────────

def extract_entities(chunk_text: str) -> ExtractionResult:
    structured_llm = llm.with_structured_output(ExtractionResult)
    result = structured_llm.invoke(f"""You are an expert entity extractor.
        Extract all meaningful entities and relationships from the given text.

        Entity types can be: Person, Technology, Project, Organization, Concept, Skill, Location, Date
        Relationship types can be: USES, BUILT, WORKED_AT, KNOWS, PART_OF, RELATED_TO, CREATED_BY

        Rules:
        - Extract only entities clearly mentioned in the text
        - Keep entity names concise and consistent
        - Only extract meaningful relationships
        - Do not hallucinate entities not present in text

        Text:
        {chunk_text}
        """)
    return result


# ── Graph Building ────────────────────────────────────────────────────────────

def chunk_for_graph(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


def build_graph_from_chunks(chunks: list) -> nx.Graph:
    g = nx.Graph()
    for chunk in chunks:
        extraction = extract_entities(chunk.page_content)

        for entity in extraction.entities:
            if not g.has_node(entity.name):
                g.add_node(entity.name, type=entity.type)

        for rel in extraction.relationships:
            if not g.has_node(rel.source):
                g.add_node(rel.source, type="Unknown")
            if not g.has_node(rel.target):
                g.add_node(rel.target, type="Unknown")
            g.add_edge(rel.source, rel.target, relation=rel.relation)

    return g


# ── Supabase ──────────────────────────────────────────────────────────────────

def save_graph_to_supabase(g: nx.Graph, user_id: str, session_id: str):
    graph_id = f"{user_id}_{session_id}"
    graph_data = nx.node_link_data(g)
    supabase.table("graph_store").upsert({
        "id": graph_id,
        "graph_data": _json.dumps(graph_data)
    }).execute()
    print(f"Graph saved: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")


def load_graph_from_supabase(user_id: str, session_id: str) -> nx.Graph:
    global graph
    graph_id = f"{user_id}_{session_id}"
    result = supabase.table("graph_store") \
        .select("graph_data") \
        .eq("id", graph_id) \
        .execute()

    if result.data:
        graph_data = _json.loads(result.data[0]["graph_data"])
        graph = nx.node_link_graph(graph_data)
        print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph

    print("No graph found in Supabase.")
    return nx.Graph()


# ── Ingestion (called by upload endpoint, not by graph node) ──────────────────

def ingest_pdf_for_graph(pdf_path: str, user_id: str, session_id: str):
    from tools.hybrid_rag_tool import load_pdf
    from pathlib import Path

    filename = Path(pdf_path).name
    docs = load_pdf(pdf_path)

    for doc in docs:
        doc.metadata["source"] = filename

    chunks = chunk_for_graph(docs)

    # Load existing graph and merge
    existing_graph = load_graph_from_supabase(user_id, session_id)
    new_graph = build_graph_from_chunks(chunks)
    merged = nx.compose(existing_graph, new_graph)
    save_graph_to_supabase(merged, user_id, session_id)

    print(f"Graph ingestion complete for {filename}")


# ── Query ─────────────────────────────────────────────────────────────────────

def query_graph(query: str, g: nx.Graph, top_k: int = 5) -> list[str]:
    if g.number_of_nodes() == 0:
        return []

    query_extraction = extract_entities(query)
    query_entities = [e.name.lower() for e in query_extraction.entities]

    matched_nodes = []
    for node in g.nodes():
        if any(qe in node.lower() or node.lower() in qe for qe in query_entities):
            matched_nodes.append(node)

    if not matched_nodes:
        degree_sorted = sorted(g.degree(), key=lambda x: x[1], reverse=True)
        matched_nodes = [n for n, d in degree_sorted[:3]]

    contexts = []
    seen = set()

    for node in matched_nodes[:top_k]:
        if node in seen:
            continue
        seen.add(node)

        neighbors = list(g.neighbors(node))
        node_type = g.nodes[node].get("type", "Unknown")

        relations = []
        for neighbor in neighbors:
            edge_data = g.edges[node, neighbor]
            relation = edge_data.get("relation", "RELATED_TO")
            neighbor_type = g.nodes[neighbor].get("type", "Unknown")
            relations.append(
                f"{node} ({node_type}) {relation} {neighbor} ({neighbor_type})"
            )

        if relations:
            context = f"[GRAPH] {node} relationships:\n" + "\n".join(relations)
            contexts.append(context)

    return contexts


# ── Graph RAG Node (query only) ───────────────────────────────────────────────

def graph_rag_node(state):
    user_query = state["messages"][-1].content
    user_id = state.get("user_id", "default")
    session_id = state.get("session_id", "default")

    # Load graph from Supabase — no file loading here
    graph_to_query = load_graph_from_supabase(user_id, session_id)

    if graph_to_query.number_of_nodes() == 0:
        print("No graph found for this session.")
        return {
            "retrieved_context": [],
            "current_context": []
        }

    contexts = query_graph(user_query, graph_to_query)
    print(f"Graph RAG found {len(contexts)} contexts.")

    return {
        "retrieved_context": contexts,
        "current_context": contexts
    }