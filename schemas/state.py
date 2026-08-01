from typing import TypedDict, List,Literal,Annotated
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    user_id : str
    session_id: str

    messages: List[BaseMessage]
    title:str
    history:List[str]
    rewritten_query:str

    #uploaded pdf 
    pdf_paths: list[str]

    #routing decision from router agent
    routing_decision: dict


    retrieved_context: Annotated[list[str], operator.add]

    #synthesizer_result
    synthesizer_result : str

    verification_attempts: int
    loop_decision: str   

    total_tokens: Annotated[int, operator.add]



from pydantic import BaseModel

class RoutingDecision(BaseModel):
    tools: list[
        Literal[
            "hybrid_rag",
            "graph_rag",
            "web_search",
            "github_rag",
            "direct"
        ]
    ]
    reasoning: str