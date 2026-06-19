from typing import TypedDict, List,Literal
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    retrieved_context: List[str]
    verification_attempts: int
    loop_decision: str   

    #uploaded pdf 
    pdf_paths: list[str]

    #routing decision from router agent
    routing_decision: dict


from pydantic import BaseModel

class RoutingDecision(BaseModel):
    tools: list[
        Literal[
            "hybrid_rag",
            "graph_rag",
            "web_search"
        ]
    ]
    reasoning: str