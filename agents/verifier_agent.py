from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal
import os

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

class VerificationResult(BaseModel):
    decision: Literal["approve", "re_evaluate"]
    reasoning: str
    issues: list[str]  

def verifier_agent(state):
    print("Verifier Agent is verifying the information...")
    
    user_query = state["messages"][-1]
    final_answer = state.get("synthesizer_result", "")
    current_context = state.get("retrieved_context", []) 
    verification_attempts = state.get("verification_attempts", 0)

    if verification_attempts >= 2:
        print("Max verification attempts reached. Approving as-is.")
        return {
            "loop_decision": "approve",
            "verification_attempts": verification_attempts,
            "total_tokens": 0
        }

    context_block = "\n\n---\n\n".join(current_context)

    structured_llm = llm.with_structured_output(VerificationResult, include_raw=True)

    raw_result = structured_llm.invoke(f"""You are an answer verifier.
    Approve if the answer is grounded in the context and attempts to address the question.
    Re-evaluate ONLY if the answer is completely hallucinated, totally off-topic, or directly contradicts the context.
    Be lenient — partial answers that are accurate should be approved.

    User Question: {user_query}
    Retrieved Context: {context_block}
    Generated Answer: {final_answer}

    Decide:""")

    result = raw_result["parsed"]        # VerificationResult pydantic object
    raw_response = raw_result["raw"]     # AIMessage with metadata

    usage = raw_response.response_metadata.get("token_usage", {})
    tokens = usage.get("total_tokens", 0)

    print(f"Verification decision: {result.decision}")
    print(f"Reasoning: {result.reasoning}")
    if result.issues:
        print(f"Issues found: {result.issues}")

    return {
        "loop_decision": result.decision,
        "verification_attempts": verification_attempts + 1,
        "total_tokens": tokens
    }