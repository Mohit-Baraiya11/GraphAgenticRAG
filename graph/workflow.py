from langgraph.graph import StateGraph,START,END
from schemas.state import AgentState
from agents.router_agent import router_agent

from tools.hybrid_rag_tool import hybrid_rag_node
from tools.graph_rag_tool import graph_rag_node
from tools.web_search_tool import web_search_node

from graph.routing import route_tools

from agents.synthesizer_agent import synthesizer_agent
from agents.verifier_agent import verifier_agent

builder = StateGraph(AgentState)

builder.add_node("router", router_agent)
builder.add_node("hybrid_rag", hybrid_rag_node)
builder.add_node("graph_rag", graph_rag_node)
builder.add_node("web_search", web_search_node)
builder.add_node("synthesizer", synthesizer_agent)
builder.add_node("verifier", verifier_agent)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_tools,
    {
        "hybrid_rag": "hybrid_rag",
        "graph_rag": "graph_rag",
        "web_search": "web_search"
    }
)
builder.add_edge("hybrid_rag", "synthesizer")
builder.add_edge("graph_rag", "synthesizer")
builder.add_edge("web_search", "synthesizer")
builder.add_edge("synthesizer","verifier")

builder.add_edge("verifier",END)

graph = builder.compile()
