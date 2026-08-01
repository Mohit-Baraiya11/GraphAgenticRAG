from langchain_groq import ChatGroq
from schemas.state import RoutingDecision
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

structured_llm = llm.with_structured_output(RoutingDecision)

def router_agent(state):
    query = state.get("rewritten_query", state["messages"][-1].content)

    recent_history = state.get("history", [])
    
        
    pdf_paths = state.get("pdf_paths", [])
    document_context = f"""
    Uploaded Documents: {pdf_paths}
    Document Count: {len(pdf_paths)}
    """
    decision = structured_llm.invoke(
        f"""
        You are a routing agent for GraphAgenticRAG system.

        Current Session Context:
        {recent_history}
        {document_context}

        Available tools:

        1. hybrid_rag
        - Use when the answer can be obtained from uploaded documents.
        - If documents are uploaded and the query appears to reference document content,
        strongly prefer hybrid_rag.

        2. graph_rag
        - Use for relationship analysis, entity connections,
        graph traversal, dependency analysis, or knowledge graphs.

        3. web_search
        - Use only when information must come from the internet,
        current events, live data, external facts, or information not available in uploaded documents.

        4. github_rag
        - Use when the user provides a GitHub repository URL (github.com/owner/repo).
        - Use for questions about code, README, project structure, bugs, architecture.
        - ALWAYS use this when message contains a github.com URL.
        - Examples:
          "explain this repo: https://github.com/user/repo"
          "find bugs in https://github.com/user/repo"
          "what does https://github.com/user/repo do?"

        5. direct
        - Use ONLY for pure greetings or small talk with zero informational content.
        - Examples: "hi", "hello", "how are you", "thanks", "bye"
        - Do NOT use if the query asks about any person, topic, skill, fact, or concept.

        Routing Guidelines:
        - If message contains github.com URL → ALWAYS use github_rag
        - Prefer hybrid_rag when documents are available
        - Use web_search only for current/external information
        - Use graph_rag only for entity relationships
        - Multiple tools may be selected if necessary

        Examples:

        User: "explain this repo: https://github.com/user/repo"
        Output: ["github_rag"]

        User: "what are the bugs in https://github.com/user/repo"
        Output: ["github_rag"]

        User: "What skills does the candidate have?"
        Output: ["hybrid_rag"]

        User: "What are the latest AI trends?"
        Output: ["web_search"]

        User Query:
        {query}
        """
        )
    usage = getattr(decision, 'response_metadata', {}).get('token_usage', {})
    tokens = usage.get('total_tokens', 0)
    return {
        "routing_decision": decision.model_dump(),
        "current_context": [],
        "total_tokens": tokens
    }

def direct_response_node(state):
    from langchain_core.messages import SystemMessage
    messages = state["messages"]
    
    full_messages = [
        SystemMessage(content="""You are GraphAgenticRAG — an intelligent AI assistant specialized in:
        - Analyzing and answering questions from uploaded PDF documents
        - Exploring and explaining GitHub repositories
        - Searching the web for current information
        - Understanding knowledge graphs and entity relationships

        You are helpful, concise, and honest. If asked what you can do, explain your capabilities.
        Do not pretend to be a generic assistant — you are GraphAgenticRAG.""")
    ] + messages
    
    result = llm.invoke(full_messages)
    usage = result.response_metadata.get("token_usage", {})
    total_tokens = usage.get("total_tokens", 0) + state.get("total_tokens", 0)
    return {
        "synthesizer_result": result.content,
        "total_tokens": total_tokens
    }