from langchain_groq import ChatGroq
from schemas.state import RoutingDecision
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

structured_llm = llm.with_structured_output(RoutingDecision)

def router_agent(state):
    query = state["messages"][-1]
    decision = structured_llm.invoke(
    f"""
    You are a routing agent.

    Available tools:

    1. hybrid_rag
    - Use ONLY when information must be retrieved from uploaded documents.

    2. graph_rag
    - Use ONLY when entity relationships, knowledge graph traversal, or graph connections are required.

    3. web_search
    - Use ONLY when current or real-time information is required.

    Rules:
    - Select the minimum number of tools required.
    - Never select all tools unless absolutely necessary.
    - Do not select document tools if no document information is requested.
    - Do not select graph_rag unless relationship analysis is needed.
    - Return only the necessary tools.

    User Query:
    {query}
    """
    )
    return {
        "routing_decision": decision.model_dump()
    }