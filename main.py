from graph.workflow import graph

message = input("Enter a message for the agent: ")
result = graph.invoke(
    {
        "pdf_paths":["Mohit_Baraiya_Resume.pdf","Mohit_cv.pdf"],
        "messages": [message],
        "retrieved_context": [],
        "verification_attempts": 0,
        "loop_decision": ""
    }
)
