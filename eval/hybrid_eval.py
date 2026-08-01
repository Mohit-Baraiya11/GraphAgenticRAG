from langchain_core.embeddings import Embeddings  
from pinecone import Pinecone
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from langchain_groq import ChatGroq
from datasets import Dataset
from langchain.messages import HumanMessage
import os
from tools.hybrid_rag_tool import ingest_pdf, hybrid_rag_node, load_bm25_from_supabase, load_parent_store_from_supabase
from agents.synthesizer_agent import synthesizer_agent
from eval.test_dataset import hybrid_rag_test_cases

PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Mohit_Baraiya_Resume.pdf")


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

class PineconeEmbeddings(Embeddings):
    def __init__(self, pc):
        self.pc = pc

    def embed_documents(self, texts):
        result = self.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=texts,
            parameters={
                "input_type": "passage",
                "truncate": "END"
            }
        )
        return [item["values"] for item in result]

    def embed_query(self, text):
        result = self.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[text],
            parameters={
                "input_type": "query",
                "truncate": "END"
            }
        )
        return result[0]["values"]
def get_ragas_embeddings():

    pinecone_embeddings = PineconeEmbeddings(pc)
    return LangchainEmbeddingsWrapper(
        pinecone_embeddings
    )

def run_retrieval_eval(user_id="eval", session_id="eval-session"):
    ingest_pdf(PDF_PATH, user_id=user_id, session_id=session_id)

    bm25_id = f"{user_id}_{session_id}"
    load_bm25_from_supabase(user_id=bm25_id)
    load_parent_store_from_supabase(user_id=user_id, session_id=session_id)

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    print(f"\nRunning {len(hybrid_rag_test_cases)} test cases...\n")

    for i,test in enumerate(hybrid_rag_test_cases):
        print(f"[{i+1}/{len(hybrid_rag_test_cases)}] {test['question'][:60]}")

        state = {
            "messages": [HumanMessage(content=test["question"])],
            "user_id": user_id,
            "session_id": session_id,
            "retrieved_context": [],
            "current_context": [],
            "routing_decision": {},
            "verification_attempts": 0,
            "loop_decision": "",
            "final_answer": "",
            "total_tokens": 0,
            "synthesizer_result": "",
            "pdf_paths": []
        }

        retrieval_result = hybrid_rag_node(state)
        contexts = retrieval_result.get("retrieved_context", [])
        state["retrieved_context"] = contexts
        synth_result = synthesizer_agent(state)
        answer = synth_result.get("synthesizer_result", "")

        questions.append(test["question"])
        answers.append(answer)
        contexts_list.append(contexts if contexts else ["No context retrieved"])
        ground_truths.append(test["ground_truth"])

        samples = []
    for q, a, c, g in zip(questions, answers, contexts_list, ground_truths):
        samples.append(SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=g
        ))
    
    dataset = EvaluationDataset(samples=samples)

    print("\nRunning RAGAS evaluation...")
    
    # New way to wrap LLM in ragas 0.2
    from ragas.llms import LangchainLLMWrapper
    from langchain_groq import ChatGroq
    ragas_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", temperature=0))
    ragas_embeddings = (
        get_ragas_embeddings()
    )
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall()
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    print(f"\n{'='*60}")
    print("RAGAS Evaluation Results")
    print(f"{'='*60}")
    print(f"Faithfulness:      {result['faithfulness']:.3f}")
    print(f"Answer Relevancy:  {result['answer_relevancy']:.3f}")
    print(f"Context Precision: {result['context_precision']:.3f}")
    print(f"Context Recall:    {result['context_recall']:.3f}")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    run_retrieval_eval()