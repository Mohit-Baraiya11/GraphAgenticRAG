from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


def synthesizer_agent(state):
    print("synthesizer agent is working ....")
    messages = state["messages"]
    current_context = state.get("retrieved_context", [])
    pdf_paths = state.get("pdf_paths", [])
    t = state.get("routing_decision", {})
    tools = t.get('tools', [])
    print(tools)

    print(f"current_context length: {len(current_context)}")

    if not current_context:
        return {
            "synthesizer_result": "I couldn't find any relevant information to answer your question."
        }

    context_block = "\n\n---\n\n".join(current_context)

    doc_note = ""
    if pdf_paths or any(tool in tools for tool in ("hybrid_rag", "graph_rag")):
        doc_note = f"Note: The user HAS uploaded {len(pdf_paths)} document(s) ({', '.join(pdf_paths) if pdf_paths else 'via retrieval'}). The context below was retrieved from it. Never say no document is attached."

    system_prompt = f"""You are an expert synthesizer. Answer the user's question using ONLY the provided context.
        {doc_note}
        Rules:
        - Use ONLY the provided context
        - If contexts conflict, mention it
        - If context is insufficient, say so
        - Do not make up information
        - Be concise and direct
        - Do NOT claim no document is attached if context is provided below — the context below is your source of truth for this turn, not prior conversation turns

        <context>
        {context_block}
        </context>"""

    recent_messages = messages[-4:]
    full_messages = [SystemMessage(content=system_prompt)] + recent_messages

    result = llm.invoke(full_messages)
    usage = result.response_metadata.get("token_usage", {})
    tokens = usage.get("total_tokens", 0) + state.get("total_tokens", 0)
    print(f"\nFinal Answer:\n{result.content}")

    return {"synthesizer_result": result.content, "total_tokens": tokens}