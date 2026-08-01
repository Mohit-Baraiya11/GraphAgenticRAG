from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

rewrite_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

REWRITE_PROMPT = """You rewrite follow-up messages into standalone questions using conversation history.

Rules:
- If the message is already a complete, standalone question/statement, return it EXACTLY as-is.
- If it depends on prior context (pronouns, "that", "nope", rebuttals, "what about..."), rewrite it to include the missing context explicitly.
- Do NOT answer the question. Only rewrite it.
- Output ONLY the rewritten text, nothing else.

Conversation History:
{history}

Message: {query}

Rewritten:"""

def query_rewriter_node(state):
    query = state["messages"][-1].content
    recent_history = state.get("history",[])
    
    if not recent_history:
        return {"rewritten_query": query}
    

    result = rewrite_llm.invoke(
        REWRITE_PROMPT.format(history=recent_history, query=query)
    )
    print(result.content)
    return {"rewritten_query": result.content.strip()}