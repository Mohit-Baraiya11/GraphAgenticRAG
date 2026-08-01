import json
from langchain_groq import ChatGroq
from schemas.state import RoutingDecision
from agents.router_agent import router_agent
from langchain_core.messages import HumanMessage
from eval.test_dataset import router_test_cases
def evaluate_router():
    results = []
    correct = 0
    total = len(router_test_cases)

    for i, test in enumerate(router_test_cases):
        # Build minimal state
        state = {
            "messages": [HumanMessage(content=test["query"])],
            "pdf_paths": ["mohit_resume.pdf"],  # simulate uploaded resume
            "user_id": "eval",
            "session_id": "eval",
            "retrieved_context": [],
            "current_context": [],
            "verification_attempts": 0,
            "loop_decision": "",
            "routing_decision": {},
            "final_answer": "",
            "total_tokens": 0,
            "synthesizer_result": "",
            "latency": {}
        }

        # Run router
        result = router_agent(state)
        actual_tools = result["routing_decision"]["tools"]
        expected_tools = test["expected"]

        # Check if correct
        is_correct = set(actual_tools) == set(expected_tools)
        if is_correct:
            correct += 1

        results.append({
            "query": test["query"],
            "expected": expected_tools,
            "actual": actual_tools,
            "correct": is_correct,
            "reason": test["reason"]
        })

        status = "✓" if is_correct else "✗"
        print(f"{status} [{i+1}/{total}] {test['query'][:50]}")
        if not is_correct:
            print(f"  Expected: {expected_tools}")
            print(f"  Got:      {actual_tools}")

    accuracy = round(correct / total * 100, 1)
    print(f"\n{'='*50}")
    print(f"Router Accuracy: {correct}/{total} = {accuracy}%")
    print(f"{'='*50}")

    # Show failures grouped
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\nFailed cases ({len(failures)}):")
        for f in failures:
            print(f"  Query: {f['query']}")
            print(f"  Expected: {f['expected']} | Got: {f['actual']}")
            print()

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
        "failures": failures
    }


if __name__ == "__main__":
    evaluate_router()