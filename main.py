from graph.workflow import graph
import uuid
from langchain_core.messages import HumanMessage
from tools.hybrid_rag_tool import ingest_pdf
from tools.graph_rag_tool import ingest_pdf_for_graph

session_id = str(uuid.uuid4())
user_id = "default"

pdf_paths = ["Mohit_Baraiya_Resume.pdf", "Mohit_cv.pdf"]

for pdf_path in pdf_paths:
    ingest_pdf(pdf_path, user_id=user_id, session_id=session_id)
    ingest_pdf_for_graph(pdf_path, user_id=user_id, session_id=session_id)

message = input("Enter a message for the agent: ")
result = graph.invoke({
    "messages": [HumanMessage(content=message)],
    "retrieved_context": [],
    "current_context": [],
    "user_id": user_id,
    "session_id": session_id,
    "verification_attempts": 0,
    "loop_decision": "",
    "routing_decision": {},
    "final_answer": ""
})