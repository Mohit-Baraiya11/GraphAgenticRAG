def route_tools(state):
    tools = state["routing_decision"]["tools"]
    print(f"Routing to: {tools}")  # ADD THIS
    return tools
def route_after_verifier(state):
    return state.get("loop_decision", "approve")
